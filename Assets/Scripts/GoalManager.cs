using UnityEngine;
using System.Collections.Generic;
using System.Collections;

public class GoalManager : BaseGoalManager
{
    // Assign a new goal by picking a random pixel from the drawn paths.
    public override void AssignNewGoal(Agent agent)
    {
        if (pathGen == null || pathGen.pathPixels.Count == 0)
        {
            Debug.LogWarning("No path pixels available. Spawning random goal.");
            AssignRandomGoal(agent);
            return;
        }

        // Convert the HashSet to a list for random access.
        List<Vector2Int> pathList = new List<Vector2Int>(pathGen.pathPixels);
        Vector2Int randomPixel = pathList[Random.Range(0, pathList.Count)];

        // Convert texture coordinate to normalized uv.
        float u = randomPixel.x / (float)pathGen.textureResolution;
        float v = randomPixel.y / (float)pathGen.textureResolution;

        // Convert uv to world position on the plane.
        // Adjust the mapping to correct the mirroring effect.
        float worldX = Mathf.Lerp(pathGen.planeSize.x / 2f, -pathGen.planeSize.x / 2f, u);
        // Invert the z-coordinate calculation to correct the mirroring.
        float worldZ = Mathf.Lerp(pathGen.planeSize.y / 2f, -pathGen.planeSize.y / 2f, v);
        Vector3 newGoal = new Vector3(worldX, goalHeight, worldZ);

        agent.goalPosition = newGoal;
        agentGoals[agent] = newGoal;
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
}
