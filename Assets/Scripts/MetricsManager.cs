using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;
using TMPro;

public class MetricsManager : MonoBehaviour
{
    [Header("UI References")]
    public TextMeshProUGUI fpsText;
    public TextMeshProUGUI efficiencyText;
    public TextMeshProUGUI civilityText;
    
    [Header("Measurement Settings")]
    public float updateInterval = 0.5f;
    
    private Simulator simulator;
    private float fpsAccumulator = 0f;
    private int fpsFrameCount = 0;
    private float currentFps = 0f;
    private float timeUntilUpdate = 0f;
    
    // Civility tracking
    private Dictionary<Agent, List<float>> agentPathComforts = new Dictionary<Agent, List<float>>();
    private Dictionary<Agent, float> agentCivilityValues = new Dictionary<Agent, float>();
    private float averageCivility = 0f;
    
    void Start()
    {
        simulator = GetComponent<Simulator>();
        timeUntilUpdate = updateInterval;
    }
    
    void Update()
    {
        // Accumulate FPS data
        fpsAccumulator += 1f / Time.deltaTime;
        fpsFrameCount++;
        
        timeUntilUpdate -= Time.deltaTime;
        if (timeUntilUpdate <= 0f)
        {
            // Calculate metrics
            CalculateFPS();
            float efficiency = CalculateEfficiency();
            float civility = CalculateCivility();
            
            // Update UI
            fpsText.text = $"FPS: {currentFps:F1}";
            efficiencyText.text = $"Efficiency: {efficiency:F3}";
            civilityText.text = $"Civility: {civility:F3}";
            
            // Reset for next update
            timeUntilUpdate = updateInterval;
        }
    }
    
    private void CalculateFPS()
    {
        currentFps = fpsAccumulator / fpsFrameCount;
        fpsAccumulator = 0f;
        fpsFrameCount = 0;
    }
    
    private float CalculateEfficiency()
    {
        int borderCellCount = 0;
        // Use a threshold of 1 pass to consider a cell part of the trail
        float passThreshold = 1f;

        for (int x = 0; x < simulator.textureResolution; x++)
        {
            for (int y = 0; y < simulator.textureResolution; y++)
            {
                // Check pass count against threshold
                if (simulator.comfortMap[x, y] >= passThreshold)
                {
                    // Check neighbors
                    bool isBorder = false; // Flag to count cell only once
                    for (int dx = -1; dx <= 1; dx++)
                    {
                        for (int dy = -1; dy <= 1; dy++)
                        {
                            if (dx == 0 && dy == 0) continue;

                            int nx = x + dx;
                            int ny = y + dy;

                            // Check bounds
                            if (nx >= 0 && nx < simulator.textureResolution &&
                                ny >= 0 && ny < simulator.textureResolution)
                            {
                                // If neighbor is below threshold, this is a border cell
                                if (simulator.comfortMap[nx, ny] < passThreshold)
                                {
                                    borderCellCount++;
                                    isBorder = true;
                                    break; // Break inner loop
                                }
                            }
                            else
                            {
                                // Edge of texture is also a border
                                borderCellCount++;
                                isBorder = true;
                                break; // Break inner loop
                            }
                        }
                        if (isBorder) break; // Break outer loop if border found
                    }
                }
            }
        }

        // Calculate efficiency (inverse of border length, normalized)
        // Higher value means less border relative to area (more efficient path)
        int totalTrailCells = 0;
        for (int x = 0; x < simulator.textureResolution; x++)
        {
            for (int y = 0; y < simulator.textureResolution; y++)
            {
                if (simulator.comfortMap[x, y] >= passThreshold)
                {
                    totalTrailCells++;
                }
            }
        }
        float efficiency = (totalTrailCells > 0 && borderCellCount > 0)
            ? (float)totalTrailCells / borderCellCount
            : 0f;
        return efficiency; // Adjust scaling if needed
    }
    
    private float CalculateCivility()
    {
        // Update civility for current agents
        foreach (Agent agent in simulator.agents)
        {
            if (agent == null) continue; // Skip if agent was destroyed

            // Get current position in texture space
            Vector2 agentTexCoord = simulator.WorldToTextureCoord(agent.transform.position);
            int x = Mathf.RoundToInt(agentTexCoord.x);
            int y = Mathf.RoundToInt(agentTexCoord.y);

            // Check bounds
            if (x >= 0 && x < simulator.textureResolution &&
                y >= 0 && y < simulator.textureResolution)
            {
                // Get comfort value (normalized pass count) at current position
                float comfort = simulator.comfortMap[x, y];
                float maxComfort = simulator.maxComfortLevel;
                float comfortValue = (maxComfort > 0)
                    ? Mathf.Clamp01(comfort / maxComfort)
                    : (comfort > 0 ? 1.0f : 0.0f); // If 0 passes needed, comfort is 1 if trail exists

                // Add to agent's path comfort list
                if (!agentPathComforts.ContainsKey(agent))
                {
                    agentPathComforts[agent] = new List<float>();
                }

                agentPathComforts[agent].Add(comfortValue);
            }
        }

        // Calculate civility for each agent
        float totalCivility = 0f;
        int agentCount = 0;

        // Use a temporary list to avoid issues if agent is destroyed during calculation
        List<Agent> currentAgents = new List<Agent>(agentPathComforts.Keys);

        foreach (Agent agent in currentAgents)
        {
             if (agent == null || !agentPathComforts.ContainsKey(agent)) continue; // Check if agent still exists

            List<float> comforts = agentPathComforts[agent];

            if (comforts.Count > 0)
            {
                float sum = 0f;
                foreach (float comfort in comforts)
                {
                    sum += comfort;
                }

                float civility = sum / comforts.Count;
                agentCivilityValues[agent] = civility;

                totalCivility += civility;
                agentCount++;
            }
        }

        // Calculate average civility
        averageCivility = agentCount > 0 ? totalCivility / agentCount : 0f;

        // Clean up for destroyed agents (safer loop)
        List<Agent> agentsToRemove = new List<Agent>();
        foreach (var agent in agentPathComforts.Keys)
        {
            // Check if the agent reference is null or the GameObject is inactive/destroyed
            if (agent == null || !simulator.agents.Contains(agent))
            {
                agentsToRemove.Add(agent);
            }
        }

        foreach (var agent in agentsToRemove)
        {
            agentPathComforts.Remove(agent);
            if (agentCivilityValues.ContainsKey(agent)) // Check before removing
            {
                agentCivilityValues.Remove(agent);
            }
        }

        return averageCivility;
    }
} 