using System.Collections;
using System.Collections.Generic;
using UnityEngine;

// Renamed from PortalManager to AgentSpawner
public class AgentSpawner : MonoBehaviour
{
    [Header("Agent Settings")]
    public GameObject agentPrefab;
    public float spawnHeight = 0.0f; // Height above portal to spawn agents
    [Tooltip("Maximum number of agents to spawn")]
    public int agentAmount = 20;
    
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
    
    [Header("Spawn Pattern")]
    [Tooltip("If true, agents will spawn in a cyclical pattern rather than randomly")]
    public bool cyclicalSpawns = false;
    
    [Header("Portal Appearance")]
    public Color spawnColor = Color.blue;
    public Color goalColor = Color.green;
    public Color dualColor = Color.cyan;
    
    [Header("Agent Management")]
    public float spawnInterval = 2f;
    [Tooltip("Distance threshold for reaching a goal")]
    public float goalReachedDistance = 1.5f;
    
    // Track which portal each agent came from
    private Dictionary<Agent, GameObject> agentOrigins = new Dictionary<Agent, GameObject>();
    private int activeAgentCount = 0;
    private int colorIndex = 0;
    
    // Indices for cyclical spawning
    private int currentSpawnIndex = 0;
    private int currentGoalIndex = 0;
    
    // Reference to simulator
    private Simulator simulator;

    // Add these fields to track agent assignments
    private Dictionary<Agent, int> agentSpawnIndices = new Dictionary<Agent, int>();
    private Dictionary<Agent, int> agentGoalIndices = new Dictionary<Agent, int>();
    private int nextSpawnIndex = 0;

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
        
    }
    
    void Update()
    {
        // Check if any agents have reached their goals
        CheckAllAgentGoals();
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
                HandleAgentReachedGoal(agent);
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
        if (simulator.agents.Count >= agentAmount)
            return null;
            
        GameObject spawnPoint;
        
        if (cyclicalSpawns)
        {
            // Get all available spawn points
            List<GameObject> allSpawnPoints = new List<GameObject>();
            allSpawnPoints.AddRange(spawnPortals);
            allSpawnPoints.AddRange(dualPortals);
            
            if (allSpawnPoints.Count == 0)
                return null;
                
            // Use the current spawn index (cycling through available spawn points)
            spawnPoint = allSpawnPoints[currentSpawnIndex % allSpawnPoints.Count];
            currentSpawnIndex++;
        }
        else
        {
            // Choose a random spawn portal (original behavior)
            List<GameObject> allSpawnPoints = new List<GameObject>();
            allSpawnPoints.AddRange(spawnPortals);
            allSpawnPoints.AddRange(dualPortals);
            
            if (allSpawnPoints.Count == 0)
                return null;
                
            spawnPoint = allSpawnPoints[Random.Range(0, allSpawnPoints.Count)];
        }
        
        // Get the exact spawn position (center of the spawner)
        Vector3 spawnPos = spawnPoint.transform.position + Vector3.up * spawnHeight;
        
        // Spawn the agent
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

        GameObject newGoalPortalGO;
        
        if (cyclicalSpawns)
        {
            // Use the current goal index (cycling through available goals)
            newGoalPortalGO = possibleDestinations[currentGoalIndex % possibleDestinations.Count];
            currentGoalIndex++;
        }
        else
        {
            // Simple logic: pick a random portal from possible destinations (original behavior)
            int attempts = 0;
            do
            {
                int randomIndex = Random.Range(0, possibleDestinations.Count);
                newGoalPortalGO = possibleDestinations[randomIndex];
                attempts++;
                // Ensure the selected portal is not null before accessing its position
                if (newGoalPortalGO == null) continue;

            } while (Vector3.Distance(newGoalPortalGO.transform.position, agent.goalPosition) < 0.1f && attempts < 10 && possibleDestinations.Count > 1); // Avoid infinite loop
        }

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

        // Ensure we have at least one color in the array
        if (agentColors == null || agentColors.Length == 0)
        {
            agentColors = new Color[] { Color.red, Color.green, Color.blue, Color.yellow, Color.magenta, Color.cyan };
        }

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
            // Create a new material instance to avoid sharing
            Material newMaterial = new Material(rend.sharedMaterial);
            newMaterial.color = assignedColor;
            rend.material = newMaterial;
        }
        
        // Set a default velocity
        agent.velocity = simulator.agentMaxSpeed_v0 * 0.5f; // Half of max speed
        
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
        if (cyclicalSpawns)
        {
            // Get all available spawn points
            List<GameObject> allSpawnPoints = new List<GameObject>();
            allSpawnPoints.AddRange(spawnPortals);
            allSpawnPoints.AddRange(dualPortals);
            
            if (allSpawnPoints.Count == 0)
                return null;
                
            // Use the current spawn index (cycling through available spawn points)
            GameObject spawnPoint = allSpawnPoints[currentSpawnIndex % allSpawnPoints.Count];
            currentSpawnIndex++;
            return spawnPoint;
        }
        else
        {
            // Choose a random spawn portal (original behavior)
            List<GameObject> allSpawnPoints = new List<GameObject>();
            allSpawnPoints.AddRange(spawnPortals);
            allSpawnPoints.AddRange(dualPortals);
            
            if (allSpawnPoints.Count == 0)
                return null;
                
            return allSpawnPoints[Random.Range(0, allSpawnPoints.Count)];
        }
    }

    // Update the SpawnAgents method to spawn agents with delay
    public List<Agent> SpawnAgents()
    {
        List<Agent> agents = new List<Agent>();
        
        // If we already have agents, don't spawn more
        if (simulator.agents.Count >= agentAmount)
        {
            Debug.Log($"Already have {simulator.agents.Count} agents, not spawning more");
            return simulator.agents;
        }
        
        Debug.Log($"Starting to spawn {agentAmount} agents with cyclicalSpawns={cyclicalSpawns}");
        
        // Start a coroutine to spawn agents with delay
        StartCoroutine(SpawnAgentsWithDelay(agents));
        
        return agents;
    }

    // Coroutine to spawn agents with delay - fixed for even distribution
    private IEnumerator SpawnAgentsWithDelay(List<Agent> agentsList)
    {
        Debug.Log($"Starting to spawn {agentAmount} agents with cyclicalSpawns={cyclicalSpawns}");
        
        // Prepare spawn and goal points
        List<GameObject> spawnPoints = new List<GameObject>();
        spawnPoints.AddRange(spawnPortals);
        spawnPoints.AddRange(dualPortals);
        
        List<GameObject> goalPoints = new List<GameObject>();
        goalPoints.AddRange(goalPortals);
        goalPoints.AddRange(dualPortals);
        
        // Check if we have valid spawn and goal points
        if (spawnPoints.Count == 0 || goalPoints.Count == 0)
        {
            Debug.LogError("No spawn or goal points available!");
            yield break;
        }
        
        // Reset indices for a fresh start
        nextSpawnIndex = 0;
        
        int agentsSpawned = 0;
        
        if (cyclicalSpawns)
        {
            // Calculate how many complete cycles we need
            int spawnPointCount = spawnPoints.Count;
            int goalPointCount = goalPoints.Count;
            
            // Create a more structured distribution pattern
            while (agentsSpawned < agentAmount)
            {
                // Get current spawn point - cycle through all spawn points first
                int spawnIdx = agentsSpawned % spawnPointCount;
                GameObject spawnPoint = spawnPoints[spawnIdx];
                
                // For each spawn point, cycle through goals
                int goalIdx = (agentsSpawned / spawnPointCount) % goalPointCount;
                GameObject goalPoint = goalPoints[goalIdx];
                
                // Skip if spawn and goal are the same
                if (spawnPoint == goalPoint)
                {
                    // Try the next goal
                    goalIdx = (goalIdx + 1) % goalPointCount;
                    goalPoint = goalPoints[goalIdx];
                    
                    // If still the same, skip this combination
                    if (spawnPoint == goalPoint && goalPointCount > 1)
                    {
                        agentsSpawned++;
                        continue;
                    }
                }
                
                // Get spawn position
                Vector3 spawnPos = spawnPoint.transform.position + Vector3.up * spawnHeight;
                
                // Spawn the agent
                Agent newAgent = SpawnSingleAgent(spawnPos);
                if (newAgent != null)
                {
                    // Set goal position
                    newAgent.goalPosition = goalPoint.transform.position;
                    
                    // Register with simulator
                    simulator.agents.Add(newAgent);
                    
                    // Track origin and assignments
                    agentOrigins[newAgent] = spawnPoint;
                    agentSpawnIndices[newAgent] = spawnIdx;
                    agentGoalIndices[newAgent] = goalIdx;
                    
                    // Set direction toward goal
                    SetAgentDirectionTowardGoal(newAgent);
                    
                    // Update counters
                    activeAgentCount++;
                    agentsList.Add(newAgent);
                    
                }
                
                // Wait before spawning next agent
                yield return new WaitForSeconds(0.1f);
                
                // Increment for next agent
                agentsSpawned++;
                
                // If we've reached the agent limit, stop spawning
                if (agentsSpawned >= agentAmount)
                    break;
            }
            
            // Update the next indices for HandleAgentReachedGoal to use
            nextSpawnIndex = 0;
        }
        else
        {
            // Original random spawning behavior
            while (agentsSpawned < agentAmount)
            {
                GameObject spawnPoint = spawnPoints[Random.Range(0, spawnPoints.Count)];
                GameObject goalPoint;
                
                // Make sure goal is different from spawn
                do {
                    goalPoint = goalPoints[Random.Range(0, goalPoints.Count)];
                } while (goalPoint == spawnPoint && goalPoints.Count > 1);
                
                // Get spawn position
                Vector3 spawnPos = spawnPoint.transform.position + Vector3.up * spawnHeight;
                
                // Spawn the agent
                Agent newAgent = SpawnSingleAgent(spawnPos);
                if (newAgent != null)
                {
                    // Set goal position
                    newAgent.goalPosition = goalPoint.transform.position;
                    
                    // Register with simulator
                    simulator.agents.Add(newAgent);
                    
                    // Track origin
                    agentOrigins[newAgent] = spawnPoint;
                    
                    // Set direction toward goal
                    SetAgentDirectionTowardGoal(newAgent);
                    
                    // Update counters
                    activeAgentCount++;
                    agentsSpawned++;
                    agentsList.Add(newAgent);
                    
                }
                
                // Wait before spawning next agent
                yield return new WaitForSeconds(0.1f);
            }
        }
        
        Debug.Log($"Finished spawning {agentsSpawned} agents");
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

    // Add this method to AgentSpawner.cs
    private void EnsureAgentColor(Agent agent)
    {
        // Check if the agent has a renderer
        Renderer rend = agent.GetComponent<Renderer>();
        if (rend == null) return;
        
        // Check if the agent's color matches the assigned color
        if (rend.material.color != agent.agentColor)
        {
            // Create a new material to avoid sharing
            Material newMaterial = new Material(Shader.Find("Standard"));
            newMaterial.color = agent.agentColor;
            rend.material = newMaterial;
            
            Debug.Log($"Fixed color for agent {agent.name}");
        }
    }

    // Update this method to use exact portal position and maintain cyclical pattern
    public void HandleAgentReachedGoal(Agent agent)
    {
        if (agent == null) return;
        
        // Get all available spawn and goal points
        List<GameObject> allSpawnPoints = new List<GameObject>();
        allSpawnPoints.AddRange(spawnPortals);
        allSpawnPoints.AddRange(dualPortals);
        
        List<GameObject> allGoalPoints = new List<GameObject>();
        allGoalPoints.AddRange(goalPortals);
        allGoalPoints.AddRange(dualPortals);
        
        if (allSpawnPoints.Count == 0 || allGoalPoints.Count == 0)
            return;
        
        GameObject spawnPoint;
        GameObject goalPoint;
        
        if (cyclicalSpawns)
        {
            // Get the next spawn point in the cycle - cycle through all spawn points first
            int spawnIdx = nextSpawnIndex % allSpawnPoints.Count;
            spawnPoint = allSpawnPoints[spawnIdx];
            
            // For each spawn point, cycle through goals
            int goalIdx = (nextSpawnIndex / allSpawnPoints.Count) % allGoalPoints.Count;
            goalPoint = allGoalPoints[goalIdx];
            
            // Skip if spawn and goal are the same
            if (spawnPoint == goalPoint && allGoalPoints.Count > 1)
            {
                goalIdx = (goalIdx + 1) % allGoalPoints.Count;
                goalPoint = allGoalPoints[goalIdx];
            }
            
            // Increment for next agent
            nextSpawnIndex++;
            
            // Store the indices for this agent
            agentSpawnIndices[agent] = spawnIdx;
            agentGoalIndices[agent] = goalIdx;
        }
        else
        {
            // Random assignment
            spawnPoint = GetRandomSpawnPortal();
            
            // Choose a random goal that's different from the spawn
            do {
                goalPoint = allGoalPoints[Random.Range(0, allGoalPoints.Count)];
            } while (goalPoint == spawnPoint && allGoalPoints.Count > 1);
        }
        
        if (spawnPoint == null) return;
        
        // Get the exact spawn position (center of the spawner)
        Vector3 newPosition = spawnPoint.transform.position + Vector3.up * spawnHeight;
        
        // Before moving the agent, completely reset its trail tracking
        if (simulator != null)
        {
            // Pass the new position to the reset method
            simulator.ResetAgentTrailTracking(agent, newPosition);
        }
        
        // Move the agent to the new spawn point
        agent.transform.position = newPosition;
        
        // Update the agent's origin
        agentOrigins[agent] = spawnPoint;
        
        // Set the goal position
        agent.goalPosition = goalPoint.transform.position;
        
        // Reset the agent's velocity toward the new goal
        SetAgentDirectionTowardGoal(agent);
        
        // Reset the agent's previous position for Verlet integration
        agent.previousPosition = newPosition;
    }

    public void SpawnAgentsWithConfig(SimulationConfig config)
    {
        // Set the vision parameters from config
        visionLength = config.visionLength;
        fieldOfView = config.fieldOfView;
        agentAmount = config.agentCount;
        
        // Spawn the agents
        SpawnAgents();
    }
} 