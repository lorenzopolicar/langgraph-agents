import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()


import uuid
from datetime import datetime, timedelta
from typing import List, Literal, TypedDict

import pandas as pd
from langchain_community.tools import DuckDuckGoSearchResults, OpenWeatherMapQueryRun
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnableSerializable,
)
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field


class Task(BaseModel):
    task_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="The unique identifier for the task",
    )
    task_name: str = Field(description="The name of the task")
    task_description: str = Field(description="A description of the task")
    task_estimated_days_to_complete: int = Field(
        description="The estimated number of days to complete the task"
    )


class TaskList(BaseModel):
    tasks: List[Task] = Field(description="A list of tasks")


class TaskDependency(BaseModel):
    task: Task = Field(description="Task")
    dependent_tasks: List[Task] = Field(description="List of dependent tasks")


class TeamMember(BaseModel):
    name: str = Field(description="The name of the team member")
    profile: str = Field(description="The profile of the team member")
    availability: str = Field(description="The availability of the team member")


class Team(BaseModel):
    members: List[TeamMember] = Field(description="A list of team members")


# Iterative assessment
class TaskAllocation(BaseModel):
    task: Task = Field(description="Task")
    team_member: TeamMember = Field(description="Team member")


class TaskSchedule(BaseModel):
    task: Task = Field(description="Task")
    start_date: datetime = Field(description="The start date of the task")
    end_date: datetime = Field(description="The end date of the task")


class DependencyList(BaseModel):
    dependencies: List[TaskDependency] = Field(description="List of task dependencies")


class Schedule(BaseModel):
    task_schedule: List[TaskSchedule] = Field(description="List of task schedules")


class TaskAllocationList(BaseModel):
    task_allocations: List[TaskAllocation] = Field(
        description="List of task allocations"
    )


class TaskAllocationListIteration(BaseModel):
    task_allocations_iteration: List[TaskAllocationList] = Field(
        description="List of task allocations for each iteration"
    )


class ScheduleIteration(BaseModel):
    schedule: List[Schedule] = Field(
        description="List of task schedules for each iteration"
    )


class Risk(BaseModel):
    task: Task = Field(description="Task")
    risk: str = Field(description="Risk associated with the task")


class RiskList(BaseModel):
    risks: List[Risk] = Field(description="List of risks")


class RiskListIteration(BaseModel):
    risks_iteration: List[RiskList] = Field(
        description="List of risks for each iteration"
    )


class AgentState(TypedDict):
    project_name: str
    project_description: str
    team: Team
    tasks: TaskList
    dependencies: DependencyList
    schedule: Schedule
    task_allocations: TaskAllocationList
    risks: RiskList
    iteration: int
    max_iterations: int
    insights: List[str]
    task_allocations_iteration: TaskAllocationListIteration
    schedule_iteration: ScheduleIteration
    risks_iteration: List[RiskListIteration]
    project_risk_score: int
    project_risk_score_iterations: List[int]


# Workflow Nodes


def task_generation_node(state: AgentState) -> AgentState:
    description = state["project_description"]
    prompt = f"""
    You are an expert project manager tasked with analyzing the following project description: {description}
        Your objectives are to: 
        1. **Extract Actionable Tasks:**
            - Identify and list all actionable and realistic tasks necessary to complete the project.
            - Provide an estimated number of days required to complete each task.
        2. **Refine Long-Term Tasks:**
            - For any task estimated to take longer than 5 days, break it down into smaller, independent sub-tasks.
        **Requirements:** - Ensure each task is clearly defined and achievable.
            - Maintain logical sequencing of tasks to facilitate smooth project execution."""

    structured_llm = llm.with_structured_output(TaskList)
    tasks: TaskList = structured_llm.invoke(prompt)

    return {"tasks": tasks}


def task_dependency_node(state: AgentState) -> AgentState:
    tasks = state["tasks"]
    prompt = f"""
    You are a skilled project scheduler responsible for mapping out task dependencies.
        Given the following list of tasks: {tasks}
        Your objectives are to:
            1. **Identify Dependencies:**
                - For each task, determine which other tasks must be completed before it can begin (blocking tasks).
            2. **Map Dependent Tasks:** 
                - For every task, list all tasks that depend on its completion.
    """
    structured_llm = llm.with_structured_output(DependencyList)
    dependencies: DependencyList = structured_llm.invoke(prompt)
    return {"dependencies": dependencies}


def task_scheduler_node(state: AgentState) -> AgentState:
    tasks = state["tasks"]
    dependencies = state["dependencies"]
    insights = state["insights"]
    prompt = f"""
        You are an experienced project scheduler tasked with creating an optimized project timeline.
        **Given:**
            - **Tasks:** {tasks}
            - **Dependencies:** {dependencies}
            - **Previous Insights:** {insights}
            - **Previous Schedule Iterations (if any):** {state["schedule_iteration"]}
        **Your objectives are to: **
            1. **Develop a Task Schedule:**
                - Assign start and end days to each task, ensuring that all dependencies are respected.
                - Optimize the schedule to minimize the overall project duration.
                - If possible parallelize the tasks to reduce the overall project duration.
                - Try not to increase the project duration compared to previous iterations.
            2. **Incorporate Insights:** 
                - Utilize insights from previous iterations to enhance scheduling efficiency and address any identified issues.
    """

    schedule_llm = llm.with_structured_output(Schedule)
    schedule: Schedule = schedule_llm.invoke(prompt)
    state["schedule"] = schedule
    state["schedule_iteration"].append(schedule)
    return state


def task_allocation_node(state: AgentState) -> AgentState:
    tasks = state["tasks"]
    schedule = state["schedule"]
    team = state["team"]
    insights = state[
        "insights"
    ]  # "" if state["insights"] is None else state["insights"].insights[-1]
    prompt = f"""
        You are a proficient project manager responsible for allocating tasks to team members efficiently.
        **Given:** 
            - **Tasks:** {tasks} 
            - **Schedule:** {schedule} 
            - **Team Members:** {team} 
            - **Previous Insights:** {insights} 
            - **Previous Task Allocations (if any):** {state["task_allocations_iteration"]} 
        **Your objectives are to:** 
            1. **Allocate Tasks:** 
                - Assign each task to a team member based on their expertise and current availability. 
                - Ensure that no team member is assigned overlapping tasks during the same time period. 
            2. **Optimize Assignments:** 
                - Utilize insights from previous iterations to improve task allocations. 
                - Balance the workload evenly among team members to enhance productivity and prevent burnout.
                **Constraints:** 
                    - Each team member can handle only one task at a time. 
                    - Assignments should respect the skills and experience of each team member.
        """
    structure_llm = llm.with_structured_output(TaskAllocationList)
    task_allocations: TaskAllocationList = structure_llm.invoke(prompt)
    state["task_allocations"] = task_allocations
    state["task_allocations_iteration"].append(task_allocations)
    return state


def risk_assessment_node(state: AgentState) -> AgentState:
    schedule = state["schedule"]
    task_allocations = state["task_allocations"]
    prompt = f"""
        You are a seasoned project risk analyst tasked with evaluating the risks associated with the current project plan.
        **Given:**
            - **Task Allocations:** {task_allocations}
            - **Schedule:** {schedule}
            - **Previous Risk Assessments (if any):** {state['risks_iteration']}
        **Your objectives are to:**
            1. **Assess Risks:**
                - Analyze each allocated task and its scheduled timeline to identify potential risks.
                - Consider factors such as task complexity, resource availability, and dependency constraints.
            2. **Assign Risk Scores:**
            - Assign a risk score to each task on a scale from 0 (no risk) to 10 (high risk).
            - If a task assignment remains unchanged from a previous iteration (same team member and task), retain the existing risk score to ensure consistency.
            - If the team member has more time between tasks - assign lower risk score for the tasks
            - If the task is assigned to a more senior person - assign lower risk score for the tasks
            3. **Calculate Overall Project Risk:**
            - Sum the individual task risk scores to determine the overall project risk score.
    """
    structured_llm = llm.with_structured_output(RiskList)
    risks: RiskList = structured_llm.invoke(prompt)
    project_risk_score = sum(risk.risk for risk in risks.risks)
    state["project_risk_score"] = project_risk_score
    state["project_risk_score_iterations"].append(project_risk_score)
    state["risks"] = risks
    state["risks_iteration"].append(risks)
    return state


def insights_generation_node(state: AgentState) -> AgentState:
    schedule = state["schedule"]
    task_allocations = state["task_allocations"]
    project_risks = state["risks"]
    prompt = f"""
        You are an expert project manager responsible for generating actionable insights to enhance the project plan.
        **Given:**
            - **Task Allocations:** {task_allocations}
            - **Schedule:** {schedule}
            - **Risk Analysis:** {project_risks}
        **Your objectives are to:**
            1. **Generate Critical Insights:**
            - Analyze the current task allocations, schedule, and risk assessments to identify areas for improvement.
            - Highlight any potential bottlenecks, resource conflicts, or high-risk tasks that may jeopardize project success.
            2. **Recommend Enhancements:**
            - Suggest adjustments to task assignments or scheduling to mitigate identified risks.
            - Propose strategies to optimize resource utilization and streamline workflow.
                **Requirements:**
                - Ensure that all recommendations aim to reduce the overall project risk score.
                - Provide clear and actionable suggestions that can be implemented in subsequent iterations.
    """
    insights = llm.invoke(prompt).content
    state["insights"].append(insights)
    return state


def router(state: AgentState) -> AgentState:
    max_iterations = state["max_iterations"]
    iteration = state["iteration"]
    if iteration < max_iterations:
        if (
            len(state["project_risk_score_iterations"]) > 1
            and state["project_risk_score_iterations"][-1]
            > state["project_risk_score_iterations"][0]
        ):
            return END
        else:
            return "insight_generator"
    else:
        return END


llm = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini",
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_5O_MINI"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

workflow = StateGraph(AgentState)
workflow.add_node("task_generator", task_generation_node)
workflow.add_node("task_dependency", task_dependency_node)
workflow.add_node("task_scheduler", task_scheduler_node)
workflow.add_node("task_allocation", task_allocation_node)
workflow.add_node("risk_assessment", risk_assessment_node)
workflow.add_node("insight_generator", insights_generation_node)


workflow.set_entry_point("task_generator")
workflow.add_edge("task_generator", "task_dependency")
workflow.add_edge("task_dependency", "task_scheduler")
workflow.add_edge("task_scheduler", "task_allocation")
workflow.add_edge("task_allocation", "risk_assessment")
workflow.add_conditional_edges("risk_assessment", router, ["insight_generator", END])
workflow.add_edge("insight_generator", "task_generator")

project_manager_agent = workflow.compile(checkpointer=MemorySaver())
project_manager_agent.name = "Project Manager Agent"
