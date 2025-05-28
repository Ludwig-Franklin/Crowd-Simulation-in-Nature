using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;
using System;
using TMPro;

public enum TestType
{
    Custom,
    I_and_T_values_for_different_trails,
    Force_and_Sigma_Values_to_match_paths,
    Sample_points_paired_with_forces_and_sigmas,
    Performance_with_varying_agent_count_and_Resolution
}

[System.Serializable]
public class TestConfiguration
{
    public string testName = "experiment_1";
    public TestType testType = TestType.Custom;
    public bool enabled = true;
}

public class ExperimentManager : MonoBehaviour
{
    [System.Serializable]
    public class SimulationConfig
    {
        public int agentCount = 30;
        public int simulationSteps = 2000;
        public bool runHelbing = false;
        public bool runVision = false;
        
        // Force Settings
        public float goalForceStrength = 10f;
        public float agentRepulsionForce = 0f;
        public float agentRepulsionRadius = 0f;
        
        // Path Following Forces
        public float pathFollowStrength = 1f;  // Helbing's method
        public float HelbingsDistanceFactor_sigma = 5f;
        public float visualPathFollowStrength = 10f;  // Vision-based method
        public float VisualDistanceFactor_sigma = 5f;
        
        // Movement Settings
        public float agentMaxSpeed_v0 = 10.0f;
        public float relaxationTime_tau = 0.1f;
        
        // Vision Sampling Settings
        public int visionArcCount = 10;
        public int firstArcPointCount = 20;
        public int lastArcPointCount = 40;
        public float fieldOfView = 120f;
        
        // Trail Settings
        public float trailWidth = 3.0f;
        public float trailRecoveryRate_T = 10f;
        public float footstepIntensity_I = 5f;
        public float maxComfortLevel = 20.0f;
        
        // Resolution
        public int textureResolution = 100;

        public string experimentName = "experiment_1";
    }

    [Header("Experiment Settings")]
    public List<SimulationConfig> simulationConfigs = new List<SimulationConfig>();
    public bool runExperiment = false;
    public int currentSimulationIndex = 0;
    public int currentMethodIndex = 0; // 0 = Helbing, 1 = Vision
    public int currentStep = 0;
    public bool experimentRunning = false;
    public bool useCyclicalSpawns = false; // Added boolean for cyclical spawns
    
    [Header("Data Collection Settings")]
    public int dataCollectionInterval = 100; // Collect data every X steps
    public bool takeScreenshots = true;
    
    [Header("References")]
    public Simulator simulator;
    public AgentSpawner agentSpawner;
    public ScriptManager scriptManager;
    public MetricsManager metricsManager;
    public Camera screenshotCamera; // Reference to the camera for screenshots

    // Data collection
    private List<float> efficiencyValues = new List<float>();
    private List<float> civilityValues = new List<float>();
    private List<float> timePerIntervalValues = new List<float>();
    private float accumulatedTime = 0f;
    private int frameCount = 1;
    private string currentSimulationName;
    private string currentExperimentFolder;
    //private int screenshotCounter = 0;
    private float intervalStartTime = 0f;
    private string currentTestName; // Add this as a class field

    [Header("UI")]
    public TextMeshProUGUI experimentStatusText;
    public UnityEngine.UI.Slider progressSlider;

    [Header("Test Configurations")]
    public List<TestConfiguration> testsToRun = new List<TestConfiguration>();

    private void Start()
    {
        // Find references if not set
        if (simulator == null) simulator = FindObjectOfType<Simulator>();
        if (agentSpawner == null) agentSpawner = FindObjectOfType<AgentSpawner>();
        if (scriptManager == null) scriptManager = FindObjectOfType<ScriptManager>();
        if (screenshotCamera == null) screenshotCamera = Camera.main;

        // Create data directory if it doesn't exist
        string baseDataPath = Path.Combine(Application.dataPath, "ExperimentData");
        if (!Directory.Exists(baseDataPath))
        {
            Directory.CreateDirectory(baseDataPath);
        }

        // Add default configurations if none exist
        if (simulationConfigs.Count == 0)
        {
            simulationConfigs.Add(new SimulationConfig());
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

    private IEnumerator RunExperiment()
    {
        experimentRunning = true;
        
        foreach (var testConfig in testsToRun)
        {
            if (!testConfig.enabled) continue;

            // Set experiment name
            currentExperimentFolder = testConfig.testName;
            
            // Create experiment folder
            currentExperimentFolder = Path.Combine(Application.dataPath, "ExperimentData", currentExperimentFolder);
            if (!Directory.Exists(currentExperimentFolder))
            {
                Directory.CreateDirectory(currentExperimentFolder);
            }

            // Setup the test configurations
            simulationConfigs.Clear();
            switch (testConfig.testType)
            {
                case TestType.Custom:
                    // Use existing configuration
                    break;
                case TestType.I_and_T_values_for_different_trails:
                    SetupTest1();
                    break;
                case TestType.Force_and_Sigma_Values_to_match_paths:
                    SetupTest2();
                    break;
                case TestType.Sample_points_paired_with_forces_and_sigmas:
                    SetupTest3();
                    break;
                case TestType.Performance_with_varying_agent_count_and_Resolution:
                    SetupTest4();
                    break;
            }

            UpdateStatusText($"Running test: {testConfig.testName}", Color.green);
            
            // Run all configurations for this test
            yield return StartCoroutine(RunAllConfigurations());
        }
        
        experimentRunning = false;
        runExperiment = false;
        UpdateStatusText("All experiments completed!", Color.green);
    }

    private IEnumerator RunAllConfigurations()
    {
        int totalSimulations = 0;
        foreach (var config in simulationConfigs)
        {
            if (config.runHelbing) totalSimulations++;
            if (config.runVision) totalSimulations++;
        }
        
        int completedSimulations = 0;
        int simulationCounter = 1;

        // Run through all simulation configs
        for (currentSimulationIndex = 0; currentSimulationIndex < simulationConfigs.Count; currentSimulationIndex++)
        {
            SimulationConfig config = simulationConfigs[currentSimulationIndex];

            if (config.runHelbing)
            {
                // Use the simulationCounter as the index in the file name
                currentSimulationName = $"{config.experimentName}_index={simulationCounter}";
                yield return StartCoroutine(RunSimulation(config, Simulator.PathFollowingMethod.HelbingsMethod, simulationCounter));
                completedSimulations++;
                simulationCounter++;
                UpdateProgress(completedSimulations, totalSimulations);
            }

            if (config.runVision)
            {
                // Use the simulationCounter as the index in the file name
                currentSimulationName = $"{config.experimentName}_index={simulationCounter}";
                yield return StartCoroutine(RunSimulation(config, Simulator.PathFollowingMethod.VisionBased, simulationCounter));
                completedSimulations++;
                simulationCounter++;
                UpdateProgress(completedSimulations, totalSimulations);
            }
        }
    }

    private IEnumerator RunSimulation(SimulationConfig config, Simulator.PathFollowingMethod method, int simulationIndex)
    {
        Debug.Log($"Starting simulation: {currentSimulationName}");
        Debug.Log($"Method: {method}, Agent Count: {config.agentCount}, Resolution: {config.textureResolution}");
        
        // Create folder structure using testName from the current test
        string baseFolder = Path.Combine(Application.dataPath, "ExperimentData"); // Use ExperimentData folder
        string testFolder = Path.Combine(baseFolder, currentTestName); // Use currentTestName
        string simulationFolder = Path.Combine(testFolder, currentSimulationName);
        
        // Create directories if they don't exist
        Directory.CreateDirectory(baseFolder);
        Directory.CreateDirectory(testFolder);
        Directory.CreateDirectory(simulationFolder);
        
        // Reset data collection
        efficiencyValues.Clear();
        civilityValues.Clear();
        timePerIntervalValues.Clear();
        accumulatedTime = 0f;
        frameCount = 1;
        intervalStartTime = Time.realtimeSinceStartup;

        // Reset the simulation
        ResetSimulation();
        
        // Apply all configuration parameters to the simulator
        ApplyConfigToSimulator(config, method);
        
        // Start the simulation
        scriptManager.StartSimulation();
        
        // Wait a frame to ensure everything is initialized
        yield return null;
        
        // Run for the specified number of steps
        for (currentStep = 0; currentStep <= config.simulationSteps; currentStep++)
        {
            // Collect data at specified intervals, but skip step 0
            if (currentStep >= dataCollectionInterval && (currentStep % dataCollectionInterval == 0 || currentStep == config.simulationSteps))
            {
                Debug.Log($"Step {currentStep}: Collecting data");
                string methodName = (method == Simulator.PathFollowingMethod.HelbingsMethod) ? "Helbing" : "Vision";
                UpdateStatusText($"{methodName}: {config.agentCount} agents - Step {currentStep}/{config.simulationSteps}", Color.green);
                
                // Calculate time since last interval
                float currentTime = Time.realtimeSinceStartup;
                float timeForInterval = currentTime - intervalStartTime;
                intervalStartTime = currentTime;
                
                // Calculate metrics
                float efficiency = CalculateEfficiency();
                float civility = CalculateCivility();
                
                // Store values
                efficiencyValues.Add(efficiency);
                civilityValues.Add(civility);
                timePerIntervalValues.Add(timeForInterval);
                
                // Take screenshot if enabled
                if (takeScreenshots)
                {
                    TakeScreenshot(simulationFolder, currentStep);
                }
                
                // Reset frame rate calculation
                accumulatedTime = 0f;
                frameCount = 1;
            }
            
            // Wait for the next fixed update
            yield return new WaitForFixedUpdate();
        }
        
        Debug.Log($"Simulation completed: {currentSimulationName}");
        // Save data for this simulation
        SaveSimulationData(config, method, simulationFolder);
        
        // Wait a moment before starting the next simulation
        yield return new WaitForSeconds(1f);
    }

    private void ApplyConfigToSimulator(SimulationConfig config, Simulator.PathFollowingMethod method)
    {
        // Set texture resolution first and reinitialize textures
        if (simulator.textureResolution != config.textureResolution)
        {
            simulator.textureResolution = config.textureResolution;
            simulator.InitializeTextures();
        }
        
        // Set the method
        simulator.currentPathFollowingMethod = method;
        
        // Set agent count
        agentSpawner.agentAmount = config.agentCount;
        
        // Set cyclical spawns
        agentSpawner.cyclicalSpawns = useCyclicalSpawns;
        
        // Apply all other parameters
        simulator.goalForceStrength = config.goalForceStrength;
        simulator.pathFollowStrength = config.pathFollowStrength;
        simulator.HelbingsDistanceFactor_sigma = config.HelbingsDistanceFactor_sigma;
        simulator.visualPathFollowStrength = config.visualPathFollowStrength;
        simulator.VisualDistanceFactor_sigma = config.VisualDistanceFactor_sigma;
        simulator.agentMaxSpeed_v0 = config.agentMaxSpeed_v0;
        simulator.relaxationTime_tau = config.relaxationTime_tau;
        simulator.trailWidth = config.trailWidth;
        simulator.trailRecoveryRate_T = config.trailRecoveryRate_T;
        simulator.maxComfortLevel = config.maxComfortLevel;
        simulator.footstepIntensity_I = config.footstepIntensity_I;
        simulator.visionArcCount = config.visionArcCount;
        simulator.firstArcPointCount = config.firstArcPointCount;
        simulator.lastArcPointCount = config.lastArcPointCount;
        
        // Set field of view for agents
        agentSpawner.fieldOfView = config.fieldOfView;
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
        simulator.agents.Clear();
        
        // Reinitialize textures and resources
        simulator.InitializeTextures();
    }

    private void TakeScreenshot(string simulationFolder, int step)
    {
        if (screenshotCamera == null) return;
        
        string filename = $"step_{step}.png";
        string filePath = Path.Combine(simulationFolder, filename);
        
        // Create a render texture
        RenderTexture rt = new RenderTexture(1920, 1080, 24);
        screenshotCamera.targetTexture = rt;
        
        // Render to the texture
        screenshotCamera.Render();
        
        // Read the render texture
        RenderTexture.active = rt;
        Texture2D screenshot = new Texture2D(1920, 1080, TextureFormat.RGB24, false);
        screenshot.ReadPixels(new Rect(0, 0, 1920, 1080), 0, 0);
        screenshot.Apply();
        
        // Reset camera
        screenshotCamera.targetTexture = null;
        RenderTexture.active = null;
        Destroy(rt);
        
        // Save to file
        byte[] bytes = screenshot.EncodeToPNG();
        File.WriteAllBytes(filePath, bytes);
        
        Destroy(screenshot);
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

    private void SaveSimulationData(SimulationConfig config, Simulator.PathFollowingMethod method, string simulationFolder)
    {
        string methodName = (method == Simulator.PathFollowingMethod.HelbingsMethod) ? "Helbing" : "Vision";
        string fileName = $"data.csv";
        string filePath = Path.Combine(simulationFolder, fileName);
        
        Debug.Log($"Saving data to: {filePath}");
        Debug.Log($"Number of data points: {efficiencyValues.Count}");
        
        StringBuilder sb = new StringBuilder();
        
        // Write configuration parameters at the top
        sb.AppendLine("# Experiment Configuration");
        sb.AppendLine($"# Method: {methodName}");
        sb.AppendLine($"# Agent Count: {config.agentCount}");
        sb.AppendLine($"# Texture Resolution: {config.textureResolution}");
        sb.AppendLine($"# Goal Force Strength: {config.goalForceStrength}");
        sb.AppendLine($"# Path Follow Strength: {config.pathFollowStrength}");
        sb.AppendLine($"# Helbing Distance Factor: {config.HelbingsDistanceFactor_sigma}");
        sb.AppendLine($"# Visual Path Follow Strength: {config.visualPathFollowStrength}");
        sb.AppendLine($"# Visual Distance Factor: {config.VisualDistanceFactor_sigma}");
        sb.AppendLine($"# Agent Max Speed: {config.agentMaxSpeed_v0}");
        sb.AppendLine($"# Relaxation Time: {config.relaxationTime_tau}");
        sb.AppendLine($"# Trail Width: {config.trailWidth}");
        sb.AppendLine($"# Trail Recovery Rate: {config.trailRecoveryRate_T}");
        sb.AppendLine($"# Max Comfort Level: {config.maxComfortLevel}");
        sb.AppendLine($"# Footstep Intensity: {config.footstepIntensity_I}");
        sb.AppendLine($"# Vision Arc Count: {config.visionArcCount}");
        sb.AppendLine($"# First Arc Point Count: {config.firstArcPointCount}");
        sb.AppendLine($"# Last Arc Point Count: {config.lastArcPointCount}");
        sb.AppendLine($"# Field of View: {config.fieldOfView}");
        sb.AppendLine($"# Simulation Steps: {config.simulationSteps}");
        sb.AppendLine($"# Data Collection Interval: {dataCollectionInterval}");
        sb.AppendLine($"# Texture Resolution: {config.textureResolution}");
        sb.AppendLine($"# Cyclical Spawns: {useCyclicalSpawns}");
        sb.AppendLine();
        
        // Write data headers
        sb.AppendLine("Step:TimeForInterval:Efficiency:Civility");
        
        // Write data rows
        for (int i = 0; i < efficiencyValues.Count; i++)
        {
            // Calculate the actual step number
            int step = (i + 1) * dataCollectionInterval;  // Start from 100, 200, etc.
            if (step > config.simulationSteps)  // If we've gone past the final step
                step = config.simulationSteps;  // Use the final step number
            
            sb.AppendLine($"{step}:{timePerIntervalValues[i]}:{efficiencyValues[i]}:{civilityValues[i]}");
        }
        
        File.WriteAllText(filePath, sb.ToString());
        Debug.Log($"Data saved successfully to: {filePath}");
    }

    private void UpdateStatusText(string text, Color color)
    {
        if (experimentStatusText != null)
        {
            experimentStatusText.text = text;
            experimentStatusText.color = color;
        }
    }

    private void UpdateProgress(int current, int total)
    {
        if (progressSlider != null)
        {
            progressSlider.value = (float)current / total;
        }
    }

    public void SetupTest1()
    {
        simulationConfigs.Clear();
        currentTestName = "1_I_and_T_values_for_different_trails";
        float[] TValues = new float[20];
        float[] IValues = new float[20];
        
        // Fill arrays with values 1 to 20
        for (int i = 0; i < 20; i++)
        {
            TValues[i] = i + 1;
            IValues[i] = i + 1;
        }
        int repetitionsPerConfig = 5;  // Number of repetitions for each configuration

        foreach (float T in TValues)
        {
            foreach (float I in IValues)
            {
                // Run each configuration multiple times
                for (int index = 1; index <= repetitionsPerConfig; index++)
                {
                    SimulationConfig config = new SimulationConfig
                    {
                        trailRecoveryRate_T = T,
                        footstepIntensity_I = I,
                        simulationSteps = 2000,
                        agentCount = 30,
                        runHelbing = true,
                        runVision = false,
                        experimentName = $"Helbing_T={T}_I={I}_repetition={index}"
                    };
                    simulationConfigs.Add(config);
                }
            }
        }
    }

    public void SetupTest2()
    {
        simulationConfigs.Clear();
        currentTestName = "2_Force_and_Sigma_Values_to_match_paths";
        float[] helbingForces = { 0.5f, 1f, 2f, 3f, 4f, 5f, 6f };
        float[] visualForces = { 60f, 80f, 100f, 120f, 140f, 160f, 180f, 200f };
        float[] sigmaValues = { 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f };
        int repetitionsPerConfig = 5;  // Number of repetitions for each configuration

        foreach (float force in helbingForces)
        {
            foreach (float sigma in sigmaValues)
            {
                // Run each configuration multiple times
                for (int index = 1; index <= repetitionsPerConfig; index++)
                {
                    SimulationConfig config = new SimulationConfig
                    {
                        pathFollowStrength = force,
                        HelbingsDistanceFactor_sigma = sigma,
                        simulationSteps = 2000,
                        agentCount = 30,
                        runHelbing = true,
                        runVision = false,
                        experimentName = $"Helbing_force={force}_sigma={sigma}_repetition={index}"
                    };
                    simulationConfigs.Add(config);
                }
            }
        }

        foreach (float force in visualForces)
        {
            foreach (float sigma in sigmaValues)
            {
                // Run each configuration multiple times
                for (int index = 1; index <= repetitionsPerConfig; index++)
                {
                    SimulationConfig config = new SimulationConfig
                    {
                        visualPathFollowStrength = force,
                        VisualDistanceFactor_sigma = sigma,
                        simulationSteps = 2000,
                        agentCount = 30,
                        runHelbing = false,
                        runVision = true,
                        experimentName = $"Vision_force={force}_sigma={sigma}_repetition={index}"
                    };
                    simulationConfigs.Add(config);
                }
            }
        }
    }

    public void SetupTest3()
    {
        simulationConfigs.Clear();
        currentTestName = "3_Sample_points_paired_with_forces_and_sigmas";
        
        // Each array contains: [arcCount, firstArc, lastArc, baseForce]
        var samplingConfigs = new (int[] config, float[] forces)[]
        {
            // Small sampling (5, 10, 20) - base force 120
            (new int[] { 5, 10, 20 }, new float[] { 100f, 110f, 120f, 130f, 140f }),
            
            // Medium sampling (10, 20, 40) - base force 30
            (new int[] { 10, 20, 40 }, new float[] { 20f, 25f, 30f, 35f, 40f }),
            
            // Large sampling (15, 30, 60) - base force 13
            (new int[] { 15, 30, 60 }, new float[] { 9f, 11f, 13f, 15f, 17f })
        };

        float[] sigmaValues = { 2f, 5f, 6f, 8f, 10f, 12f, 15f, 20f };
        int repetitionsPerConfig = 5;  // Number of repetitions for each configuration

        // For each sampling configuration
        foreach (var (config, forces) in samplingConfigs)
        {
            // For each force variation
            foreach (float force in forces)
            {
                // For each sigma value
                foreach (float sigma in sigmaValues)
                {
                    // Run each configuration multiple times
                    for (int index = 1; index <= repetitionsPerConfig; index++)
                    {
                        SimulationConfig simConfig = new SimulationConfig
                        {
                            visionArcCount = config[0],
                            firstArcPointCount = config[1],
                            lastArcPointCount = config[2],
                            visualPathFollowStrength = force,
                            VisualDistanceFactor_sigma = sigma,
                            simulationSteps = 2000,
                            agentCount = 30,
                            runHelbing = false,
                            runVision = true,
                            experimentName = $"Vision_arcs={config[0]}_first={config[1]}_last={config[2]}_force={force}_sigma={sigma}_repetition={index}"
                        };
                        simulationConfigs.Add(simConfig);
                    }
                }
            }
        }
    }

    public void SetupTest4()
    {
        simulationConfigs.Clear();
        currentTestName = "4_Performance_with_varying_agent_count_and_Resolution";
        int[] resolutions = { 100, 200, 300};
        int[] agentCounts = { 1, 10, 20, 30, 40};
        int repetitionsPerConfig = 3;

        foreach (int resolution in resolutions)
        {
            foreach (int agentCount in agentCounts)
            {
                // Run each configuration multiple times
                for (int index = 1; index <= repetitionsPerConfig; index++)
                {
                    // Helbing's method config
                    SimulationConfig helbingConfig = new SimulationConfig
                    {
                        goalForceStrength = 10f,
                        pathFollowStrength = 2f,
                        HelbingsDistanceFactor_sigma = 7f,
                        textureResolution = resolution,
                        simulationSteps = 1000,
                        agentCount = agentCount,
                        runHelbing = true,
                        runVision = false,
                        experimentName = $"Helbing_resolution={resolution}_agents={agentCount}_repetition={index}"
                    };
                    simulationConfigs.Add(helbingConfig);

                    // Vision-based method config
                    SimulationConfig visionConfig = new SimulationConfig
                    {
                        goalForceStrength = 10f,
                        visionArcCount = 5,
                        firstArcPointCount = 10,
                        lastArcPointCount = 20,
                        visualPathFollowStrength = 180f,
                        VisualDistanceFactor_sigma = 4f,
                        textureResolution = resolution,
                        simulationSteps = 1000,
                        agentCount = agentCount,
                        runHelbing = false,
                        runVision = true,
                        experimentName = $"Vision_resolution={resolution}_agents={agentCount}_repetition={index}"
                    };
                    simulationConfigs.Add(visionConfig);
                }
            }
        }
    }
} 