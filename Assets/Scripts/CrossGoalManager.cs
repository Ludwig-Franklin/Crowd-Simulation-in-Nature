using UnityEngine;
using System.Collections.Generic;
using System.Collections;

public class CrossGoalManager : BaseGoalManager
{
    // Enum to track the current goal position in the clockwise sequence
    private enum GoalPosition { Top, Right, Bottom, Left }
    
    // Dictionary to track each agent's current goal position in the sequence
    private Dictionary<Agent, GoalPosition> agentGoalPositions = new Dictionary<Agent, GoalPosition>();
    
    // Margin from the edge of the plane
    public float edgeMargin = 1.0f;

    public override void AssignNewGoal(Agent agent)
    {
        if (pathGen == null)
        {
            Debug.LogWarning("No path generator available. Spawning random goal.");
            AssignRandomGoal(agent);
            return;
        }

        // Get or initialize the agent's current position in the sequence
        if (!agentGoalPositions.ContainsKey(agent))
        {
            agentGoalPositions[agent] = GoalPosition.Top;
        }
        else
        {
            // Move to the next position in the clockwise sequence
            agentGoalPositions[agent] = (GoalPosition)(((int)agentGoalPositions[agent] + 1) % 4);
        }

        // Get the current goal position
        GoalPosition currentPosition = agentGoalPositions[agent];
        
        // Calculate the world position based on the current goal position
        Vector3 newGoal = CalculateGoalPosition(currentPosition);
        
        agent.goalPosition = newGoal;
        agentGoals[agent] = newGoal;
    }

    private Vector3 CalculateGoalPosition(GoalPosition position)
    {
        float halfWidth = pathGen.planeSize.x / 2f - edgeMargin;
        float halfHeight = pathGen.planeSize.y / 2f - edgeMargin;
        
        switch (position)
        {
            case GoalPosition.Top:
                // Top of the vertical line
                return new Vector3(0, goalHeight, halfHeight);
                
            case GoalPosition.Right:
                // Right end of the horizontal line
                return new Vector3(halfWidth, goalHeight, 0);
                
            case GoalPosition.Bottom:
                // Bottom of the vertical line
                return new Vector3(0, goalHeight, -halfHeight);
                
            case GoalPosition.Left:
                // Left end of the horizontal line
                return new Vector3(-halfWidth, goalHeight, 0);
                
            default:
                return Vector3.zero;
        }
    }

    // Fallback: assign a random goal within plane bounds.
    protected void AssignRandomGoal(Agent agent)
    {
        Vector2 planeSize = pathGen.planeSize;
        float randomX = Random.Range(-planeSize.x / 2f, planeSize.x / 2f);
        float randomZ = Random.Range(-planeSize.y / 2f, planeSize.y / 2f);
        Vector3 newGoal = new Vector3(randomX, goalHeight, randomZ);
        agent.goalPosition = newGoal;
        agentGoals[agent] = newGoal;
    }

    protected override void OnDrawGizmos()
    {
        base.OnDrawGizmos();
        
        if (pathGen == null) return;
        
        // Draw the cross pattern goal positions for debugging
        Gizmos.color = Color.cyan;
        float halfWidth = pathGen.planeSize.x / 2f - edgeMargin;
        float halfHeight = pathGen.planeSize.y / 2f - edgeMargin;
        
        // Top
        Gizmos.DrawWireSphere(new Vector3(0, goalHeight, halfHeight), 0.3f);
        // Right
        Gizmos.DrawWireSphere(new Vector3(halfWidth, goalHeight, 0), 0.3f);
        // Bottom
        Gizmos.DrawWireSphere(new Vector3(0, goalHeight, -halfHeight), 0.3f);
        // Left
        Gizmos.DrawWireSphere(new Vector3(-halfWidth, goalHeight, 0), 0.3f);
    }
} 