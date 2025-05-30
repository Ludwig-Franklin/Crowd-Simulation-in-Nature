using UnityEngine;

[System.Serializable]
public class SimulationConfig
{
    // Basic simulation settings
    [Header("Basic Settings")]
    public string experimentName = "experiment_1";
    public int agentCount = 30;
    public int simulationSteps = 2000;
    public bool runHelbing = false;
    public bool runVision = false;
    public int textureResolution = 100;
    
    [Header("Force Settings")]
    public float goalForceStrength = 10f;
    public float agentRepulsionForce = 0f;
    public float agentRepulsionRadius = 0f;
    public float pathFollowStrength = 2f;  // Helbing's method
    public float HelbingsDistanceFactor_sigma = 7f;
    public float visualPathFollowStrength = 20f;  // Vision-based method
    public float VisualDistanceFactor_sigma = 10f;
    
    [Header("Movement Settings")]
    public float agentMaxSpeed_v0 = 15.0f;
    public float relaxationTime_tau = 0.1f;
    
    [Header("Vision Settings")]
    public int visionArcCount = 8;
    public int firstArcPointCount = 20;
    public int lastArcPointCount = 40;
    public float visionLength = 80;
    public float fieldOfView = 180f;
    
    [Header("Trail Settings")]
    public float trailWidth = 3.0f;
    public float trailRecoveryRate_T = 10f;
    public float footstepIntensity_I = 5f;
    public float maxComfortLevel = 20.0f;

    // Add a constructor to ensure all fields are properly initialized
    public SimulationConfig()
    {
        // Constructor initializes all fields with default values
        // This helps ensure proper serialization
    }
} 