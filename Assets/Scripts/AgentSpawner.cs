using System.Collections.Generic;
using UnityEngine;

// Renamed from PortalManager to AgentSpawner
public class AgentSpawner : MonoBehaviour
{
    [Header("Agent Settings")]
    public GameObject agentPrefab;
    public float spawnHeight = 0.0f; // Height above portal to spawn agents
    public int amountOfAgents = 1;
    
    // Pre-defined array of colors to assign
    public Color[] agentColors = new Color[] { Color.red, Color.green, Color.blue, Color.yellow, Color.magenta, Color.cyan };
    
    [Header("Vision Settings")]
    [Tooltip("Vision length for agents")]
    public float visionLength = 5f;
    
    [Tooltip("Field of view angle in degrees")]
    public float fieldOfView = 60f;
    
    [Header("Portal Settings")]
    public List<GameObject> spawnPortals = new List<GameObject>();
    public List<GameObject> goalPortals = new List<GameObject>();
    public List<GameObject> dualPortals = new List<GameObject>(); // Both spawn and goal
    
    [Header("Portal Appearance")]
    public Color spawnColor = Color.blue;
    public Color goalColor = Color.green;
    public Color dualColor = Color.cyan;
    
    [Header("Agent Management")]
    public int maxAgents = 20;
    public float spawnInterval = 2f;
    [Tooltip("Distance threshold for reaching a goal")]
    public float goalReachedDistance = 1.5f;
    
    // Track which portal each agent came from
    private Dictionary<Agent, GameObject> agentOrigins = new Dictionary<Agent, GameObject>();
    private float nextSpawnTime = 0f;
    private int activeAgentCount = 0;
    private int colorIndex = 0;
    
    // Reference to simulator
    private Simulator simulator;
    
    void Start()
    {
        // Find the simulator
        simulator = FindObjectOfType<Simulator>();
        if (simulator == null)
        {
            Debug.LogError("Simulator not found in the scene!");
            return;
        }
        
        // Set up portal colors
        SetupPortalColors();
        
        // Initial spawn of agents up to maxAgents
        for (int i = 0; i < maxAgents && i < 5; i++) // Start with at most 5 agents
        {
            SpawnNewAgent();
        }
    }
    
    void Update()
    {
        // Check if we need to spawn more agents to maintain the desired count
        if (simulator.agents.Count < maxAgents)
        {
            // Calculate how many agents we need to spawn
            int agentsToSpawn = maxAgents - simulator.agents.Count;
            
            // Spawn new agents up to the maximum
            for (int i = 0; i < agentsToSpawn; i++)
            {
                if (Time.time > nextSpawnTime)
                {
                    SpawnNewAgent();
                    nextSpawnTime = Time.time + spawnInterval;
                }
            }
        }
    }
    
    private void CheckAllAgentGoals()
    {
        // Create a copy of the agents list to avoid modification during iteration
        List<Agent> agentsToCheck = new List<Agent>(simulator.agents);
        
        foreach (Agent agent in agentsToCheck)
        {
            if (agent == null) continue;
            
            // Check if agent has reached its goal
            float distanceToGoal = Vector3.Distance(agent.transform.position, agent.goalPosition);
            
            if (distanceToGoal < goalReachedDistance)
            {
                // Remove the agent from simulator's list
                simulator.RemoveAgent(agent);
                
                // Remove from our tracking
                agentOrigins.Remove(agent);
                
                // Destroy the agent
                Destroy(agent.gameObject);
                
                // Spawn a new agent to replace this one
                SpawnNewAgent();
            }
        }
    }
    
    private void SetupPortalColors()
    {
        // Set colors for spawn portals
        foreach (var portal in spawnPortals)
        {
            if (portal != null)
                SetPortalColor(portal, spawnColor);
        }
        
        // Set colors for goal portals
        foreach (var portal in goalPortals)
        {
            if (portal != null)
                SetPortalColor(portal, goalColor);
        }
        
        // Set colors for dual portals
        foreach (var portal in dualPortals)
        {
            if (portal != null)
                SetPortalColor(portal, dualColor);
        }
    }
    
    private void SetPortalColor(GameObject portal, Color color)
    {
        Renderer renderer = portal.GetComponent<Renderer>();
        if (renderer != null && renderer.material != null)
        {
            // Check if the material is already set to avoid creating duplicates
            if (renderer.material.color != color)
            {
                renderer.material.color = color;
            }
        }
    }
    
    private Agent SpawnNewAgent()
    {
        // Don't spawn if we're at max capacity
        if (simulator.agents.Count >= maxAgents)
            return null;
            
        // Choose a random spawn portal
        List<GameObject> allSpawnPoints = new List<GameObject>();
        allSpawnPoints.AddRange(spawnPortals);
        allSpawnPoints.AddRange(dualPortals);
        
        if (allSpawnPoints.Count == 0)
            return null;
            
        GameObject spawnPoint = allSpawnPoints[Random.Range(0, allSpawnPoints.Count)];
        
        // Spawn the agent
        Vector3 spawnPos = spawnPoint.transform.position + Vector3.up * spawnHeight;
        Agent agent = SpawnSingleAgent(spawnPos);
        
        // Register the agent with the simulator
        simulator.agents.Add(agent);
        
        // Track where this agent came from
        agentOrigins[agent] = spawnPoint;
        
        // Assign a goal
        AssignNewGoal(agent);
        
        // Set the agent's initial direction toward the goal
        SetAgentDirectionTowardGoal(agent);
        
        activeAgentCount++;
        return agent;
    }
    
    public void AssignNewGoal(Agent agent)
    {
        // Combine goal and dual portals into a list of possible destinations
        List<GameObject> possibleDestinations = new List<GameObject>();
        possibleDestinations.AddRange(goalPortals);
        possibleDestinations.AddRange(dualPortals);

        if (possibleDestinations.Count == 0)
        {
            Debug.LogError("AgentSpawner: No goal or dual portals assigned!");
            return;
        }

        // Log Current Goal
        Vector3 oldGoal = agent.goalPosition;

        // Simple logic: pick a random portal from possible destinations
        GameObject newGoalPortalGO = null;
        int attempts = 0;
        do
        {
            int randomIndex = Random.Range(0, possibleDestinations.Count);
            newGoalPortalGO = possibleDestinations[randomIndex];
            attempts++;
            // Ensure the selected portal is not null before accessing its position
            if (newGoalPortalGO == null) continue;

        } while (Vector3.Distance(newGoalPortalGO.transform.position, agent.goalPosition) < 0.1f && attempts < 10 && possibleDestinations.Count > 1); // Avoid infinite loop

        if (newGoalPortalGO != null)
        {
            agent.goalPosition = newGoalPortalGO.transform.position;
            
            // Set the agent's direction toward the new goal
            SetAgentDirectionTowardGoal(agent);
        }
        else
        {
            Debug.LogWarning($"Agent {agent.name}: Could not find a different portal for a new goal.");
        }
    }
    
    // Moved from AgentSpawner class
    public Agent SpawnSingleAgent(Vector3 position)
    {
        Agent agent = Instantiate(agentPrefab, position, Quaternion.identity).GetComponent<Agent>();
        
        // Set the agent's radius (assumes agent's prefab scale represents its size)
        agent.radius = agentPrefab.transform.localScale.x / 2;

        // Assign a unique color from the array (cycling through if there are more agents than colors)
        Color assignedColor = agentColors[colorIndex % agentColors.Length];
        colorIndex++;
        agent.agentColor = assignedColor;
        agent.fixedY = spawnHeight;
        
        // Set vision parameters from the spawner settings
        agent.visionLength = visionLength;
        agent.fieldOfView = fieldOfView;

        // Update the agent's material color
        Renderer rend = agent.GetComponent<Renderer>();
        if (rend != null)
        {
            rend.material = new Material(rend.material);
            rend.material.color = assignedColor;
        }
        
        // Set a default velocity
        agent.velocity = simulator.agentMaxSpeed * 0.5f; // Half of max speed
        
        // Give the agent a unique name
        agent.name = "Agent_" + activeAgentCount;
        
        return agent;
    }
    
    // Helper method to get a random goal portal
    private GameObject GetRandomGoalPortal()
    {
        // Get all possible goal portals
        List<GameObject> possibleGoals = new List<GameObject>();
        possibleGoals.AddRange(goalPortals);
        possibleGoals.AddRange(dualPortals);
        
        if (possibleGoals.Count == 0)
            return null;
            
        // Choose a random goal
        return possibleGoals[Random.Range(0, possibleGoals.Count)];
    }

    // Helper method to get a random spawn portal
    private GameObject GetRandomSpawnPortal()
    {
        // Choose a random spawn portal
        List<GameObject> allSpawnPoints = new List<GameObject>();
        allSpawnPoints.AddRange(spawnPortals);
        allSpawnPoints.AddRange(dualPortals);
        
        if (allSpawnPoints.Count == 0)
            return null;
            
        return allSpawnPoints[Random.Range(0, allSpawnPoints.Count)];
    }

    // Add this method to the AgentSpawner class
    public List<Agent> SpawnAgents()
    {
        List<Agent> agents = new List<Agent>();
        // Spawn the specified number of agents
        for (int i = 0; i < amountOfAgents; i++)
        {
            Agent newAgent = SpawnNewAgent();
            agents.Add(newAgent);
        }
        return agents;
    }

    // Add this method to set the agent's initial direction toward its goal
    private void SetAgentDirectionTowardGoal(Agent agent)
    {
        if (agent == null || agent.goalPosition == Vector3.zero)
            return;
        
        // Calculate direction from agent to goal
        Vector3 direction = agent.goalPosition - agent.transform.position;
        
        // Ensure we're only moving in the XZ plane
        direction.y = 0;
        
        // Normalize the direction
        if (direction.magnitude > 0.001f)
        {
            direction.Normalize();
            agent.currentDirection = direction;
            
            // Optionally rotate the agent's transform to face the goal
            if (direction != Vector3.zero)
            {
                agent.transform.forward = direction;
            }
        }
    }
} 