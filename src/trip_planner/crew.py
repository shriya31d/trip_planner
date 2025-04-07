import os
from crewai import Agent, Crew, Process, Task,LLM
from crewai.project import CrewBase, agent, crew, task

# Uncomment the following line to use an example of a custom tool
# from surprise_travel.tools.custom_tool import MyCustomTool

# Check our tools documentation for more information on how to use them
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from pydantic import BaseModel, Field
from typing import List, Optional

from src.pydantic_models.TravelItinerary import TravelItinerary
from src.tools.custom_tool import add_actual_dates_in_itinerary

@CrewBase
class TripPlannerCrew():
    """TripPlanner crew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    manager_llm = LLM(
        model="azure/gpt-4o",
        base_url= os.getenv("AZURE_API_BASE"),
        api_key= os.getenv("AZURE_API_KEY")
    )

    @agent
    def sightseeing_planner(self) -> Agent:
        return Agent(
            config=self.agents_config['sightseeing_planner'],
            tools=[SerperDevTool(n_results=5), ScrapeWebsiteTool()], # Example of custom tool, loaded at the beginning of file
            verbose=True,
            allow_delegation=False,
        )

    @agent
    def itinerary_compiler(self) -> Agent:
        return Agent(
            config=self.agents_config['itinerary_compiler'],
            tools=[add_actual_dates_in_itinerary],
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
            verbose = True,
            process=Process.sequential
        )