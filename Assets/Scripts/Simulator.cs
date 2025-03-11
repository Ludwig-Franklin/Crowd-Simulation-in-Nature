using UnityEngine;
using System.Collections.Generic;

public class Simulator : MonoBehaviour
{
    [Header("Movement Settings")]
    public float drivingForce = 10f;
    public float repulsionForce = 2f;
    public float repulsionRadius = 1.5f;
    public float goalDistanceThreshold = 0.5f;
    public float maxSpeed = 3f;

    [Header("Path Following Settings")]
    [Tooltip("Increase this value to make agents follow paths more strongly.")]
    public float pathFollowStrength = 10f;
    [Tooltip("Maximum multiplier for path following when far from path")]
    public float maxPathDistanceFactor = 10f;
    [Tooltip("Distance in pixels at which path following reaches maximum strength")]
    public float maxPathDistanceThreshold = 50f;
    [Tooltip("Controls how quickly path following strength increases with distance (higher = faster)")]
    public float pathDistanceCurve = 3f;
    [Range(-1f, 1f)]
    [Tooltip("Controls how much the path force can deviate from the goal direction (-1: any direction, 0: 90° max, 1: only toward goal)")]
    public float maxAngleFromGoalForce = 0f;

    [Header("References")]
    public BasePathGenerator pathGen;
    private PaintPlane paintPlane;

    public List<Agent> agents = new List<Agent>();
    public BaseGoalManager goalManager;
    public Vector2 textureSize;

    public GradientGenerator gradientGenerator;

    [Header("Debug Visualization")]
    public bool showForces = true;
    public float forceLineScale = 1f;
    public Color goalForceColor = Color.green;
    public Color avoidanceForceColor = Color.red;
    public Color pathForceColor = Color.blue;
    public Color totalForceColor = Color.yellow;

    private List<List<float>> distanceMap;
    private Vector2Int[] directions;
    private float maxDistance;

    void Awake()
    {
        goalManager = GetComponent<BaseGoalManager>();
        if (pathGen == null)
            pathGen = GetComponent<BasePathGenerator>();
        paintPlane = GetComponent<PaintPlane>();
        gradientGenerator = GetComponent<GradientGenerator>();

        textureSize = new Vector2(pathGen.textureResolution, pathGen.textureResolution);
        
        directions = new Vector2Int[] {
            new Vector2Int(0, 1),
            new Vector2Int(1, 0),
            new Vector2Int(0, -1),
            new Vector2Int(-1, 0),
            new Vector2Int(1, 1),
            new Vector2Int(1, -1),
            new Vector2Int(-1, 1),
            new Vector2Int(-1, -1)
        };
    }

    void FixedUpdate()
    {
        if (distanceMap == null)
        {
            distanceMap = gradientGenerator.distanceMap;
            maxDistance = gradientGenerator.maxDistance;
        }
        foreach (Agent agent in agents)
        {
            if (!agent.gameObject.activeSelf)
                continue;

            Vector3 goalForce = CalculateGoalForce(agent);
            Vector3 avoidanceForce = CalculateAvoidanceForce(agent);
            Vector3 pathForce = CalculatePathForce(agent);

            Vector3 totalForce = goalForce + avoidanceForce + pathForce;
            totalForce.y = 0;  // Keep movement horizontal

            // Visualize forces if enabled
            if (showForces)
            {
                // Position slightly above the agent to make forces visible
                Vector3 startPos = agent.transform.position + Vector3.up * 0.1f;
                
                // Draw individual forces
                Debug.DrawRay(startPos, goalForce * forceLineScale, goalForceColor);
                Debug.DrawRay(startPos, avoidanceForce * forceLineScale, avoidanceForceColor);
                Debug.DrawRay(startPos, pathForce * forceLineScale, pathForceColor);
                
                // Draw resulting force
                Debug.DrawRay(startPos, totalForce.normalized * forceLineScale, totalForceColor);
            }

            agent.velocity = Mathf.Clamp(totalForce.magnitude, 0, maxSpeed);
            agent.transform.position += totalForce.normalized * agent.velocity * Time.fixedDeltaTime;
            // Lock y position.
            agent.transform.position = new Vector3(agent.transform.position.x, agent.fixedY, agent.transform.position.z);
            agent.transform.rotation = Quaternion.identity;
        }
    }

    Vector3 CalculateGoalForce(Agent agent)
    {
        Vector3 toGoal = agent.goalPosition - agent.transform.position;
        if (toGoal.magnitude < goalDistanceThreshold)
        {
            goalManager.AssignNewGoal(agent);
            toGoal = agent.goalPosition - agent.transform.position;
        }
        return toGoal.normalized * drivingForce;
    }

    Vector3 CalculateAvoidanceForce(Agent agent)
    {
        Vector3 avoidance = Vector3.zero;
        foreach (Agent other in agents)
        {
            if (other == agent || !other.gameObject.activeSelf)
                continue;

            Vector3 direction = agent.transform.position - other.transform.position;
            float distance = direction.magnitude;
            if (distance < repulsionRadius)
            {
                float strength = Mathf.Clamp01(1 - (distance / repulsionRadius));
                avoidance += direction.normalized * strength * repulsionForce;
            }
        }
        return avoidance;
    }

    Vector3 CalculatePathForce(Agent agent)
    {
        // Check if distanceMap is null before using it
        if (distanceMap == null)
        {
            Debug.LogWarning("Distance map is null in CalculatePathForce");
            return Vector3.zero;
        }

        Vector2 texCoord = WorldToTextureCoord(agent.transform.position);
        int x = (int)texCoord.x;
        int y = (int)texCoord.y;

        // Check array bounds before accessing
        if (x < 0 || x >= distanceMap.Count || y < 0 || y >= distanceMap[x].Count || distanceMap[x] == null)
        {
            Debug.LogError($"Out of bounds access: x={x}, y={y}, distanceMap[x]={distanceMap[x]}");
            return Vector3.zero;
        }

        if (distanceMap[x][y] < 2)
        {
            Debug.Log("Agent is on the track");
            return Vector3.zero;
        }

        float minNeighborDistance = float.MaxValue;
        Vector2Int bestDirection = new Vector2Int(0, 0);
        float maxGoalAlignment = -1f; // -1 is the worst possible alignment (opposite direction)

        // Calculate normalized direction to goal
        Vector3 goalDirection = (agent.goalPosition - agent.transform.position).normalized;

        foreach (Vector2Int direction in directions)
        {
            int neighborX = x + direction.x;
            int neighborY = y + direction.y;

            if (neighborX >= 0 && neighborX < pathGen.textureResolution &&
                neighborY >= 0 && neighborY < pathGen.textureResolution)
            {
                float neighborDistance = distanceMap[neighborX][neighborY];
                
                // Calculate direction vector and its alignment with goal direction
                Vector3 directionVector = new Vector3(direction.x, 0, direction.y).normalized;
                float goalAlignment = Vector3.Dot(directionVector, goalDirection);

                // First prioritize getting closer to the path, then consider goal alignment
                if ((neighborDistance < minNeighborDistance && goalAlignment >= maxAngleFromGoalForce) || 
                    (Mathf.Approximately(neighborDistance, minNeighborDistance) && goalAlignment > maxGoalAlignment))
                {
                    minNeighborDistance = neighborDistance;
                    bestDirection = direction;
                    maxGoalAlignment = goalAlignment;
                }
            }
        }

        // Configurable exponential distance factor
        float normalizedDistance = Mathf.Min(1.0f, distanceMap[x][y] / maxPathDistanceThreshold);
        float distanceFactor = 1.0f + (maxPathDistanceFactor - 1.0f) * (1.0f - Mathf.Exp(-pathDistanceCurve * normalizedDistance));
        
        return (new Vector3(bestDirection.x, 0, bestDirection.y)).normalized 
            * distanceFactor
            * pathFollowStrength;
    }

    Vector2 WorldToTextureCoord(Vector3 worldPos)
    {
        Vector3 localPos = pathGen.transform.InverseTransformPoint(worldPos);
        float u = Mathf.Clamp01((localPos.x + pathGen.planeSize.x / 2) / pathGen.planeSize.x);
        float v = Mathf.Clamp01((localPos.z + pathGen.planeSize.y / 2) / pathGen.planeSize.y);
        return new Vector2(u * (pathGen.textureResolution - 1), v * (pathGen.textureResolution - 1));
    }

    public void RegisterAgents(List<Agent> newAgents)
    {
        agents.AddRange(newAgents);
    }
}
