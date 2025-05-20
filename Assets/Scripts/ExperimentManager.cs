using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;
using System;
using TMPro;

public class ExperimentManager : MonoBehaviour
{
    [System.Serializable]
    public class SimulationConfig
    {
        public int agentCount;
        public int simulationSteps;
        public bool runHelbing = true;
        public bool runVision = true;
    }

    [Header("Experiment Settings")]
    public List<SimulationConfig> simulationConfigs = new List<SimulationConfig>();
    public bool runExperiment = false;
    public int currentSimulationIndex = 0;
    public int currentMethodIndex = 0; // 0 = Helbing, 1 = Vision
    public int currentStep = 0;
    public bool experimentRunning = false;
    public string dataFolderName = "simulation_data";

    [Header("References")]
    public Simulator simulator;
    public AgentSpawner agentSpawner;
    public ScriptManager scriptManager;
    public MetricsManager metricsManager;

    // Data collection
    private List<float> efficiencyValues = new List<float>();
    private List<float> civilityValues = new List<float>();
    private List<float> frameRates = new List<float>();
    private float accumulatedTime = 0f;
    private int frameCount = 0;
    private string currentSimulationName;

    [Header("UI")]
    public TextMeshProUGUI experimentStatusText;
    public UnityEngine.UI.Slider progressSlider;

    private void Start()
    {
        // Find references if not set
        if (simulator == null) simulator = FindObjectOfType<Simulator>();
        if (agentSpawner == null) agentSpawner = FindObjectOfType<AgentSpawner>();
        if (scriptManager == null) scriptManager = FindObjectOfType<ScriptManager>();

        // Create data directory if it doesn't exist
        string path = Path.Combine(Application.dataPath, dataFolderName);
        if (!Directory.Exists(path))
        {
            Directory.CreateDirectory(path);
        }

        // Add default configurations if none exist
        if (simulationConfigs.Count == 0)
        {
            simulationConfigs.Add(new SimulationConfig { agentCount = 10, simulationSteps = 1000 });
            simulationConfigs.Add(new SimulationConfig { agentCount = 20, simulationSteps = 1000 });
            simulationConfigs.Add(new SimulationConfig { agentCount = 50, simulationSteps = 1000 });
            simulationConfigs.Add(new SimulationConfig { agentCount = 100, simulationSteps = 1000 });
        }
    }

    private void Update()
    {
        if (runExperiment && !experimentRunning)
        {
            StartCoroutine(RunExperiment());
        }

        // Collect frame rate data during experiment
        if (experimentRunning)
        {
            accumulatedTime += Time.deltaTime;
            frameCount++;
        }
    }

    public IEnumerator RunExperiment()
    {
        experimentRunning = true;
        currentSimulationName = "simulation_" + GetNextSimulationNumber();
        UpdateStatusText($"Starting experiment: {currentSimulationName}", Color.green);
        
        int totalSimulations = 0;
        foreach (var config in simulationConfigs)
        {
            if (config.runHelbing) totalSimulations++;
            if (config.runVision) totalSimulations++;
        }
        
        int completedSimulations = 0;

        // Run through all simulation configs
        for (currentSimulationIndex = 0; currentSimulationIndex < simulationConfigs.Count; currentSimulationIndex++)
        {
            SimulationConfig config = simulationConfigs[currentSimulationIndex];
            UpdateStatusText($"Running simulation with {config.agentCount} agents for {config.simulationSteps} steps", Color.green);

            // Run Helbing method if configured
            if (config.runHelbing)
            {
                UpdateStatusText($"Method: Helbing, Agents: {config.agentCount}", Color.green);
                yield return StartCoroutine(RunSimulation(config, Simulator.PathFollowingMethod.HelbingsMethod));
                completedSimulations++;
                UpdateProgress(completedSimulations, totalSimulations);
            }

            // Run Vision method if configured
            if (config.runVision)
            {
                UpdateStatusText($"Method: Vision, Agents: {config.agentCount}", Color.green);
                yield return StartCoroutine(RunSimulation(config, Simulator.PathFollowingMethod.VisionBased));
                completedSimulations++;
                UpdateProgress(completedSimulations, totalSimulations);
            }
        }

        // Save all data
        SaveExperimentData();
        
        experimentRunning = false;
        runExperiment = false;
        UpdateStatusText("Experiment completed!", Color.red);
        Debug.Log("Experiment completed!");
    }

    private IEnumerator RunSimulation(SimulationConfig config, Simulator.PathFollowingMethod method)
    {
        // Reset data collection
        efficiencyValues.Clear();
        civilityValues.Clear();
        frameRates.Clear();
        accumulatedTime = 0f;
        frameCount = 0;

        // Reset the simulation
        ResetSimulation();
        
        // Set the method
        simulator.currentPathFollowingMethod = method;
        
        // Set agent count
        agentSpawner.agentAmount = config.agentCount;
        
        // Start the simulation
        scriptManager.StartSimulation();
        
        // Wait a frame to ensure everything is initialized
        yield return null;
        
        // Run for the specified number of steps
        for (currentStep = 0; currentStep < config.simulationSteps; currentStep++)
        {
            // Update status every 100 steps
            if (currentStep % 100 == 0)
            {
                string methodName = (method == Simulator.PathFollowingMethod.HelbingsMethod) ? "Helbing" : "Vision";
                UpdateStatusText($"{methodName}: {config.agentCount} agents - Step {currentStep}/{config.simulationSteps}", Color.green);
                
                // Calculate efficiency and civility
                float efficiency = CalculateEfficiency();
                float civility = CalculateCivility();
                
                // Calculate average frame rate
                float frameRate = frameCount > 0 ? frameCount / accumulatedTime : 0;
                
                // Store values
                efficiencyValues.Add(efficiency);
                civilityValues.Add(civility);
                frameRates.Add(frameRate);
                
                // Reset frame rate calculation
                accumulatedTime = 0f;
                frameCount = 0;
                
                Debug.Log($"Step {currentStep}: Efficiency = {efficiency}, Civility = {civility}, FPS = {frameRate}");
            }
            
            // Wait for the next fixed update
            yield return new WaitForFixedUpdate();
        }
        
        // Save data for this simulation
        SaveSimulationData(config, method);
        
        // Wait a moment before starting the next simulation
        yield return new WaitForSeconds(1f);
    }

    private void ResetSimulation()
    {
        // Clear existing agents
        foreach (var agent in simulator.agents.ToArray())
        {
            if (agent != null)
            {
                simulator.RemoveAgent(agent);
                Destroy(agent.gameObject);
            }
        }
        
        // Reset comfort map
        if (simulator.comfortMap != null)
        {
            for (int x = 0; x < simulator.textureResolution; x++)
            {
                for (int y = 0; y < simulator.textureResolution; y++)
                {
                    simulator.comfortMap[x, y] = 0f;
                }
            }
        }
        
        // Force texture update
        simulator.forceTextureUpdate = true;
    }

    private float CalculateEfficiency()
    {
        if (metricsManager != null)
        {
            return metricsManager.GetEfficiency();
        }
        
        // Fallback to original implementation
        float borderSum = 0f;
        int resolution = simulator.textureResolution;
        float intensityThreshold = 0.1f; // Adjust as needed
        
        for (int x = 1; x < resolution - 1; x++)
        {
            for (int y = 1; y < resolution - 1; y++)
            {
                float G = simulator.comfortMap[x, y];
                if (G > intensityThreshold)
                {
                    // Check neighbors
                    for (int dx = -1; dx <= 1; dx++)
                    {
                        for (int dy = -1; dy <= 1; dy++)
                        {
                            if (dx == 0 && dy == 0) continue;
                            
                            int nx = x + dx;
                            int ny = y + dy;
                            
                            if (nx >= 0 && nx < resolution && ny >= 0 && ny < resolution)
                            {
                                float neighborG = simulator.comfortMap[nx, ny];
                                if (neighborG < intensityThreshold)
                                {
                                    borderSum += 1.0f;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Normalize by a constant k
        float k = 0.01f; // Adjust as needed
        float efficiency = (borderSum > 0) ? (k / borderSum) : 0f;
        
        return efficiency;
    }

    private float CalculateCivility()
    {
        if (metricsManager != null)
        {
            return metricsManager.GetCivility();
        }
        
        // Fallback to original implementation
        if (simulator.agents.Count == 0) return 0f;
        
        float totalCivility = 0f;
        
        foreach (Agent agent in simulator.agents)
        {
            if (agent == null) continue;
            
            // Get agent's position in texture coordinates
            Vector2 texCoord = simulator.WorldToTextureCoord(agent.transform.position);
            int texX = Mathf.RoundToInt(texCoord.x);
            int texY = Mathf.RoundToInt(texCoord.y);
            
            // Get comfort value at agent's position
            float comfort = 0f;
            if (texX >= 0 && texX < simulator.textureResolution && 
                texY >= 0 && texY < simulator.textureResolution)
            {
                comfort = simulator.comfortMap[texX, texY];
            }
            
            totalCivility += comfort;
        }
        
        return totalCivility / simulator.agents.Count;
    }

    private void SaveSimulationData(SimulationConfig config, Simulator.PathFollowingMethod method)
    {
        string methodName = (method == Simulator.PathFollowingMethod.HelbingsMethod) ? "Helbing" : "Vision";
        string fileName = $"{currentSimulationName}_{config.agentCount}agents_{methodName}.csv";
        string path = Path.Combine(Application.dataPath, dataFolderName, fileName);
        
        StringBuilder sb = new StringBuilder();
        sb.AppendLine("Step,Efficiency,Civility,FrameRate");
        
        for (int i = 0; i < efficiencyValues.Count; i++)
        {
            int step = i * 100;
            if (i == efficiencyValues.Count - 1) step = config.simulationSteps - 1;
            
            sb.AppendLine($"{step},{efficiencyValues[i]},{civilityValues[i]},{frameRates[i]}");
        }
        
        File.WriteAllText(path, sb.ToString());
        Debug.Log($"Saved simulation data to {path}");
    }

    private void SaveExperimentData()
    {
        string fileName = $"{currentSimulationName}_summary.csv";
        string path = Path.Combine(Application.dataPath, dataFolderName, fileName);
        
        StringBuilder sb = new StringBuilder();
        sb.AppendLine("AgentCount,Method,AverageEfficiency,AverageCivility,AverageFrameRate");
        
        // We'll need to parse the individual files to create the summary
        string dataPath = Path.Combine(Application.dataPath, dataFolderName);
        string[] files = Directory.GetFiles(dataPath, $"{currentSimulationName}_*.csv");
        
        foreach (string file in files)
        {
            if (file.Contains("summary")) continue;
            
            string filename = Path.GetFileNameWithoutExtension(file);
            string[] parts = filename.Split('_');
            
            if (parts.Length >= 3)
            {
                string agentCountStr = parts[1].Replace("agents", "");
                string method = parts[2];
                
                int agentCount;
                if (int.TryParse(agentCountStr, out agentCount))
                {
                    // Read the file and calculate averages
                    string[] lines = File.ReadAllLines(file);
                    float sumEfficiency = 0f;
                    float sumCivility = 0f;
                    float sumFrameRate = 0f;
                    int count = 0;
                    
                    for (int i = 1; i < lines.Length; i++) // Skip header
                    {
                        string[] values = lines[i].Split(',');
                        if (values.Length >= 4)
                        {
                            float efficiency, civility, frameRate;
                            if (float.TryParse(values[1], out efficiency) &&
                                float.TryParse(values[2], out civility) &&
                                float.TryParse(values[3], out frameRate))
                            {
                                sumEfficiency += efficiency;
                                sumCivility += civility;
                                sumFrameRate += frameRate;
                                count++;
                            }
                        }
                    }
                    
                    if (count > 0)
                    {
                        float avgEfficiency = sumEfficiency / count;
                        float avgCivility = sumCivility / count;
                        float avgFrameRate = sumFrameRate / count;
                        
                        sb.AppendLine($"{agentCount},{method},{avgEfficiency},{avgCivility},{avgFrameRate}");
                    }
                }
            }
        }
        
        File.WriteAllText(path, sb.ToString());
        Debug.Log($"Saved experiment summary to {path}");
    }

    private int GetNextSimulationNumber()
    {
        string path = Path.Combine(Application.dataPath, dataFolderName);
        if (!Directory.Exists(path))
        {
            return 1;
        }
        
        string[] files = Directory.GetFiles(path, "simulation_*_summary.csv");
        int maxNumber = 0;
        
        foreach (string file in files)
        {
            string filename = Path.GetFileNameWithoutExtension(file);
            string[] parts = filename.Split('_');
            
            if (parts.Length >= 2)
            {
                int number;
                if (int.TryParse(parts[1], out number))
                {
                    maxNumber = Mathf.Max(maxNumber, number);
                }
            }
        }
        
        return maxNumber + 1;
    }

    private void UpdateStatusText(string text, Color color)
    {
        if (experimentStatusText != null)
        {
            experimentStatusText.text = text;
            experimentStatusText.color = color;
        }
        Debug.Log(text);
    }

    private void UpdateProgress(int current, int total)
    {
        if (progressSlider != null)
        {
            progressSlider.value = (float)current / total;
        }
    }
} 