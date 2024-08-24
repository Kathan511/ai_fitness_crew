from crewai import Agent, Task, Crew


# use of tagging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from crewai_tools import (
  FileReadTool,
  ScrapeWebsiteTool,
  MDXSearchTool,
  SerperDevTool,
  YoutubeChannelSearchTool
)

search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

class Fitness(BaseModel):
    main_goal: str = Field(description="The fitness goal or to achive or focused body part to improve(i.e. Chest,Abs,etc.). If not provided set it to `overall body` ")
    expected_time: str = Field(
        description="The provided exptected time. If not provided set it to `no duration`"
    )
    location: str = Field(description="The location to do fitness excercises. If the location is not provided then set it to `any location`")
    equipment: str = Field(description="The equipement to use for excercise.If the equipment is not provided then set it to `any gym_equipments`")





def extract_tags(user_input:str) -> dict:
    tagging_prompt = ChatPromptTemplate.from_template(
    """
    Extract the desired information from the following passage.

    Only extract the properties mentioned in the 'Classification' function.

    Passage:
    {input}
    """
    )
    # LLM
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo").with_structured_output(
    Fitness)
    tagging_chain = tagging_prompt | llm
    output=tagging_chain.invoke({"input": user_input})

    return output.dict()




def execute_crew(user_tags:dict):
    fitness_coach_agent=Agent(
    role='Fitness coach',
    goal="Make sure to give clear and effective"
        "list of exercises guide and repetitions that align with client's goals"
        "If the client's requirements are unrealistic, then"
        "don't give false answers.",
    verbose=True,
    backstory=(
        "As an experienced Fitness coach, you have all the knowledge"
        "about all types of exercises. You help "
        "clients by suggesting the list of exercises that "
        "aligns with their fitness goals."
    )    
    )


    content_writer_agent=Agent(
    role="Professional content writer",
        goal="Restructure the given text that aligns with proper format.",
        backstory="You are professional content writes, who writes "
                "articles about fitness."
    )


    # Task for Researcher Agent: Extract Job Requirements
    fitness_task = Task(
        description=(
            "Suggest fitness exercises "
            " for {main_goal} in {expected_time} time at {location}."
            " If required, use the tools to give brief answers to exercises."
        ),
        expected_output=(
            "A structured list of names of exercises, with steps to do it."
            " Also include the repetitions of a particular exercise."
        ),
        agent=fitness_coach_agent,
        tools=[scrape_tool,search_tool]

    )

    # Task for Researcher Agent: Extract Job Requirements
    content_writer_task = Task(
        description=(
            "Give well-written fitness exercises with description and repetitions"
            "related article with focused on {main_goal} at {location} for {expected_time} with {equipment} "
        ),
        expected_output=(
            "A well-written blog post in markdown format"),
        agent=content_writer_agent
    )

    fitness_coach_crew = Crew(
    agents=[fitness_coach_agent,content_writer_agent],

    tasks=[fitness_task,content_writer_task],
    verbose=True)

    markdown_result = fitness_coach_crew.kickoff(inputs=user_tags)

    return markdown_result