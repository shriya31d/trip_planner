from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

# Uncomment the following line to use an example of a custom tool
# from surprise_travel.tools.custom_tool import MyCustomTool

# Check our tools documentation for more information on how to use them
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from pydantic import BaseModel, Field
from typing import List, Optional

class TravelItinerary(BaseModel):
    destination: str = Field(..., description="Travel destination (e.g., Kashmir, India)")
    start_date: str = Field(..., description="Start date of the trip in YYYY-MM-DD format")
    duration_days: int = Field(..., description="Total number of days for the trip")
    traveler_age: Optional[int] = Field(None, description="Age of the traveler for personalization")
    season: Optional[str] = Field(None, description="Inferred season based on the start date and destination")
    day: int = Field(..., description="Day number of the itinerary")
    date: Optional[str] = Field(None, description="Date corresponding to the day, optional if auto-calculated")
    spots: List[str] = Field(..., description="List of sightseeing spots for the day")
    location: str = Field(..., description="Primary location or area visited on this day (e.g., Srinagar, Gulmarg, Pahalgam)")
    
@CrewBase
class TripPlannerCrew():
    """TripPlanner crew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def sightseeing_planner(self) -> Agent:
        return Agent(
            config=self.agents_config['sightseeing_planner'],
            tools=[SerperDevTool(), ScrapeWebsiteTool()], # Example of custom tool, loaded at the beginning of file
            verbose=True,
            allow_delegation=False,
        )

    @agent
    def itinerary_compiler(self) -> Agent:
        return Agent(
            config=self.agents_config['itinerary_compiler'],
            tools=[SerperDevTool()],
            verbose=True,
            allow_delegation=False,
        )

    @task
    def sigthseeing_planning_task(self) -> Task:
        return Task(
            config=self.tasks_config['sightseeing_planning_task'],
            agent=self.sightseeing_planner()
        )

    @task
    def itinerary_compilation_task(self) -> Task:
        return Task(
            config=self.tasks_config['itinerary_compilation_task'],
            agent=self.itinerary_compiler(),
            output_json=TravelItinerary
        )

    @crew
    def crew(self) -> Crew:
        """Creates the TripPlanner crew"""
        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose = True,
            # process=Process.hierarchical, # In case you want to use that instead https://docs.crewai.com/how-to/Hierarchical/
        )