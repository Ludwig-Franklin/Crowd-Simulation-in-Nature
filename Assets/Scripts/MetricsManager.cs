using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;
using TMPro;
using System.Linq;

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
        const float k = 1000f;  // Normalization constant
        int borderCount = 0;

        // For each cell in the grid
        for (int x = 1; x < simulator.textureResolution - 1; x++)
        {
            for (int y = 1; y < simulator.textureResolution - 1; y++)
            {
                float G = simulator.comfortMap[x, y];
                float I = simulator.footstepIntensity_I;

                // If this is a path cell (G > I)
                if (G > I)
                {
                    // Check its neighbors
                    for (int dx = -1; dx <= 1; dx++)
                    {
                        for (int dy = -1; dy <= 1; dy++)
                        {
                            if (dx == 0 && dy == 0) continue;
                            
                            float neighborG = simulator.comfortMap[x + dx, y + dy];
                            // Count neighbors where G < I (border cells)
                            if (neighborG < I)
                            {
                                borderCount++;
                            }
                        }
                    }
                }
            }
        }

        // Efficiency is inverse of border length, normalized by k
        // If there are no borders (no paths), efficiency should be 0
        return borderCount > 0 ? k / borderCount : 0f;
    }
    
    private float CalculateCivility()
    {
        // Update civility for current agents
        foreach (Agent agent in simulator.agents)
        {
            if (agent == null) continue;

            // Get current position in texture space
            Vector2 agentTexCoord = simulator.WorldToTextureCoord(agent.transform.position);
            int x = Mathf.RoundToInt(agentTexCoord.x);
            int y = Mathf.RoundToInt(agentTexCoord.y);

            // Check bounds
            if (x >= 0 && x < simulator.textureResolution &&
                y >= 0 && y < simulator.textureResolution)
            {
                // Get raw comfort value at current position
                float comfort = simulator.comfortMap[x, y];

                // Add to agent's path comfort list
                if (!agentPathComforts.ContainsKey(agent))
                {
                    agentPathComforts[agent] = new List<float>();
                }

                agentPathComforts[agent].Add(comfort);  // Store raw value
            }
        }

        // Calculate civility for each agent
        float totalCivility = 0f;
        int agentCount = 0;

        foreach (Agent agent in new List<Agent>(agentPathComforts.Keys))
        {
            if (agent == null || !agentPathComforts.ContainsKey(agent)) continue;

            List<float> comforts = agentPathComforts[agent];
            if (comforts.Count > 0)
            {
                // Simple average of raw comfort values
                float civility = comforts.Sum() / comforts.Count;
                agentCivilityValues[agent] = civility;
                totalCivility += civility;
                agentCount++;
            }
        }

        averageCivility = agentCount > 0 ? totalCivility / agentCount : 0f;
        return averageCivility;
    }

    // Add these methods to expose metrics
    public float GetEfficiency()
    {
        return CalculateEfficiency();
    }

    public float GetCivility()
    {
        return CalculateCivility();
    }

    public float GetCurrentFPS()
    {
        return currentFps;
    }
} 