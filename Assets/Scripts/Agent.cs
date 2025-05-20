using UnityEngine;
using System.Collections.Generic;

public class Agent : MonoBehaviour
{
    public float velocity;
    public float radius;
    public Vector3 goalPosition;
    // The agent's unique color.
    public Color agentColor = Color.white;

    public float fixedY;
    
    // Sight visualization settings
    [Header("Sight Visualization")]
    [Tooltip("Show the sight box in the editor")]
    public bool showSightBox = true;
    [Tooltip("Color of the sight box")]
    public Color sightBoxColor = Color.yellow;
    
    [Tooltip("Color of sampling points")]
    public Color samplingPointColor = Color.yellow;
    [Tooltip("Color for points with trails")]
    public Color trailPointColor = Color.green;
    [Tooltip("Size for the sampled points")]
    public float pointSize = 0.5f;


    // Vision cone parameters
    [Header("Vision Cone Settings")]
    [Tooltip("Length of the agent's vision cone")]
    public float visionLength = 10f;
    [Tooltip("Field of view angle in degrees")]
    public float fieldOfView = 120f;

    [Header("Debug Visualization")]
    public Color goalForceColor = Color.green;
    public Color avoidanceForceColor = Color.red;
    public Color pathForceColor = Color.blue; // For gradient following
    public Color trailVisionForceColor = Color.magenta; // For vision following
    public Color totalForceColor = Color.yellow;
    public float forceScale = 1f;

    // --- Add fields to store calculated forces for visualization ---
    [HideInInspector] public Vector3 lastGoalForce;
    [HideInInspector] public Vector3 lastAvoidanceForce;
    [HideInInspector] public Vector3 lastPathForce; // Includes gradient or vision force
    [HideInInspector] public Vector3 lastTotalForce;
    // -------------------------------------------------------------

    // Reference to simulator for coordinate conversion
    private Simulator simulator;

    // Current movement direction (normalized)
    [HideInInspector]
    public Vector3 currentDirection = Vector3.forward;
    
    // List of sample points for visualization
    [HideInInspector]
    public List<SamplePoint> samplePoints = new List<SamplePoint>();

    // Add this field to store the previous position for Verlet integration
    [HideInInspector] public Vector3 previousPosition;

    void Start()
    {
        // Find simulator only once
        simulator = FindObjectOfType<Simulator>();
        if (simulator == null)
        {
            Debug.LogError("Simulator not found in the scene!", this);
        }

        // Initialize previousPosition
        previousPosition = transform.position;

        // --- Add Check for Initial Direction ---
        if (currentDirection == Vector3.zero)
        {
            // If somehow it's still zero, assign a random XZ direction
            currentDirection = Random.onUnitSphere;
            currentDirection.y = 0;
            currentDirection.Normalize();
            Debug.LogWarning($"Agent {name} had zero initial direction, setting to random: {currentDirection}");
        }
        // --------------------------------------

        // Set the material color
        Renderer rend = GetComponent<Renderer>();
        if (rend != null)
        {
            rend.material = new Material(rend.material); // Create instance
            rend.material.color = agentColor;
        }
    }

    // OnDrawGizmos runs for all agents if Gizmos are enabled
    void OnDrawGizmos()
    {
        // Find simulator if needed
        if (simulator == null)
        {
            simulator = FindObjectOfType<Simulator>();
            if (simulator == null) return;
        }

        // --- Draw Forces --- (Keep this here to see forces for all agents)
        if (simulator.showForces) // Check the global flag
        {
            DrawForceGizmos(); // Call the helper method
        }
    }

    // OnDrawGizmosSelected runs ONLY for the agent selected in the Hierarchy
    void OnDrawGizmosSelected()
    {
        // Find simulator if needed
        if (simulator == null)
        {
            simulator = FindObjectOfType<Simulator>();
            if (simulator == null) return;
        }

        // --- Draw Vision Cone ---
        if (showSightBox && simulator.showSampledPoints)
        {
            Gizmos.color = sightBoxColor;
            
            // Ensure forward direction is in XZ plane
            Vector3 forward = currentDirection.normalized;
            forward.y = 0;
            if (forward.magnitude < 0.01f) forward = Vector3.forward;
            forward.Normalize();
            
            // Calculate right vector (perpendicular to forward in XZ plane)
            Vector3 right = Vector3.Cross(Vector3.up, forward).normalized;
            
            float halfAngle = fieldOfView / 2f * Mathf.Deg2Rad;
            
            // Calculate left and right rays using proper XZ plane rotation
            Vector3 leftRay = forward * Mathf.Cos(-halfAngle) + right * Mathf.Sin(-halfAngle);
            Vector3 rightRay = forward * Mathf.Cos(halfAngle) + right * Mathf.Sin(halfAngle);
            
            Gizmos.DrawRay(transform.position, leftRay * visionLength);
            Gizmos.DrawRay(transform.position, rightRay * visionLength);
            
            int segments = 20;
            Vector3 prevPoint = transform.position + leftRay * visionLength;
            
            for (int i = 1; i <= segments; i++)
            {
                float angle = -halfAngle + (i * (2 * halfAngle) / segments);
                Vector3 direction = forward * Mathf.Cos(angle) + right * Mathf.Sin(angle);
                Vector3 point = transform.position + direction * visionLength;
                Gizmos.DrawLine(prevPoint, point);
                prevPoint = point;
            }
        }

        // --- Draw Sample Points ---
        // Draw sample points ONLY if the global flag is set
        if (simulator.showSampledPoints)
        {
            // Always get fresh sample points when selected in the editor
            List<SamplePoint> points = simulator.GetSamplePointsForAgent(this);
            if (points != null)
            {
                foreach (var point in points)
                {
                    if (point.hasTrail)
                    {
                        Gizmos.color = trailPointColor;
                        Gizmos.DrawSphere(point.position, pointSize);
                    }
                    else
                    {
                        Gizmos.color = samplingPointColor;
                        Gizmos.DrawSphere(point.position, pointSize);
                    }
                }
            }
        }
    }

    // Helper method to draw forces (called from OnDrawGizmos)
    void DrawForceGizmos()
    {
        Vector3 origin = transform.position;
        float scale = forceScale;

        // Goal Force (Green)
        if (lastGoalForce.sqrMagnitude > 0.01f)
        {
            Gizmos.color = goalForceColor;
            Gizmos.DrawRay(origin, lastGoalForce.normalized * scale); // Draw normalized * scale
        }

        // Avoidance Force (Red)
        if (lastAvoidanceForce.sqrMagnitude > 0.01f)
        {
            Gizmos.color = avoidanceForceColor;
            Gizmos.DrawRay(origin, lastAvoidanceForce.normalized * scale);
        }

        // Path/Vision Force (Blue or Magenta depending on method)
        if (lastPathForce.sqrMagnitude > 0.01f)
        {
             if (simulator.currentPathFollowingMethod == Simulator.PathFollowingMethod.VisionBased)
             {
                  Gizmos.color = trailVisionForceColor; // Magenta for vision
             }
             else
             {
                  Gizmos.color = pathForceColor; // Blue for gradient
             }
             Gizmos.DrawRay(origin, lastPathForce.normalized * scale);
        }


        // Total Force (Yellow) - Draw this last so it's potentially on top
        if (lastTotalForce.sqrMagnitude > 0.01f)
        {
            Gizmos.color = totalForceColor;
            Gizmos.DrawRay(origin, lastTotalForce.normalized * scale);
        }
    }
}
