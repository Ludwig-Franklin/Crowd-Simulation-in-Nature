using UnityEngine;
using System.Collections.Generic;
using Unity.Jobs;
using Unity.Collections;
using System.Linq;

public class Simulator : MonoBehaviour
{
    // --- Define the Enum ---
    public enum PathFollowingMethod
    {
        HelbingsMethod, // Helbing's method
        VisionBased    // Gaze-driven method (uses local vision)
    }
    // -----------------------

    [Header("Force Settings")]
    [Tooltip("Strength of the force pulling agents toward their goals")]
    public float goalForceStrength = 10f;

    [Tooltip("Strength of repulsion between agents")]
    public float agentRepulsionForce = 0f;

    [Tooltip("Radius within which agents repel each other")]
    public float agentRepulsionRadius = 0f;

    [Header("Path Following Forces")]
    [Tooltip("Strength of Helbing's path following force")]
    public float pathFollowStrength = 10f;

    [Tooltip("Distance factor for Helbing's method (higher values reduce effect at distance)")]
    [Range(0f, 10f)]
    public float HelbingsDistanceFactor_sigma = 0.1f;

    [Tooltip("Strength of vision-based path following force")]
    public float visualPathFollowStrength = 10f;

    [Tooltip("Distance factor for vision-based method (higher values reduce effect at distance) (sigma)")]
    [Range(0f, 10f)]
    public float VisualDistanceFactor_sigma = 0.1f;

    [Header("Movement Limits")]
    [Tooltip("Maximum speed agents can move")]
    public float agentMaxSpeed_v0 = 3f;

    [Tooltip("Time for agents to adjust to desired velocity")]
    public float relaxationTime_tau = 1f;

    [Tooltip("Distance at which agents are considered to have reached their goal")]
    public float goalReachedThreshold = 0.5f;

    [Header("Movement Settings")]
    [Tooltip("Select the method agents use to follow trails.")]
    public PathFollowingMethod currentPathFollowingMethod = PathFollowingMethod.HelbingsMethod; // Default method

    [Header("Vision Sampling Settings")]
    [Tooltip("Number of arcs to sample from agent to vision range")]
    [Range(3, 1000)]
    public int visionArcCount = 8;

    [Tooltip("Number of points on the first arc (closest to agent)")]
    [Range(3, 500)]
    public int firstArcPointCount = 5;

    [Tooltip("Number of points on the last arc (farthest from agent)")]
    [Range(5, 1000)]
    public int lastArcPointCount = 15;
    
    [Header("Debug Visualization")]
    public bool showForces = false;
    public bool showSampledPoints = false; // Keep this flag
    
    [Header("Path Creation Settings")]
    [Tooltip("Width of the trail in pixels")]
    public float trailWidth = 6f;

    [Header("Trail Settings")]
    [Tooltip("Time scale for trail decay in seconds (T in equation 5) - higher values mean slower decay")]
    [Range(0.0f, 10.0f)]
    public float trailRecoveryRate_T = 0.05f; // T in equation (5)

    [Header("Comfort Map Settings")]
    [Tooltip("Maximum comfort level a trail can reach (Gmax)")]
    public float maxComfortLevel = 20f; // Gmax in equation (5)

    [Tooltip("Intensity of each footstep (I) - between 0 and 1")]
    [Range(0.0f, 1.0f)]
    public float footstepIntensity_I = 0.8f; // I in equation (5)

    // Color gradient for comfort visualization
    public Color comfortColor0 = Color.red;     // 0
    public Color comfortColor1 = Color.yellow;  // ~Gmax * 0.2
    public Color comfortColor2 = Color.green;   // ~Gmax * 0.4
    public Color comfortColor3 = Color.cyan;    // ~Gmax * 0.6
    public Color comfortColor4 = Color.blue;    // ~Gmax * 0.8
    public Color comfortColor5 = Color.magenta; // ~Gmax * 0.9
    public Color comfortColor6 = Color.white;   // Gmax

    // Comfort map data
    public float[,] comfortMap; // Stores comfort values from 0 to Gmax

    [Header("References")]
    public AgentSpawner agentSpawner; // Changed from goalManager
    public ComputeShader trailComputeShader;
    public Material planeMaterial; // Add reference to the plane's material
    
    [Header("Texture Settings")]
    public int textureResolution = 256;

    // Private variables
    private Color[] trailColors;
    public bool forceTextureUpdate = false;
    private ComputeBuffer trailPositionBuffer;
    private Dictionary<Agent, int> agentLastTrailUpdateFrame = new Dictionary<Agent, int>();
    private Dictionary<Agent, List<SamplePoint>> agentSamplePoints = new Dictionary<Agent, List<SamplePoint>>();
    private Dictionary<Agent, Vector2> agentPreviousPositions = new Dictionary<Agent, Vector2>();
    private RenderTexture trailTexture;
    private bool[,] hasTrailMap;
    private bool[,] changedPixels;
    private List<Vector2Int> changedPixelsList = new List<Vector2Int>();
    private HashSet<Vector2Int> steppedCellsThisFrame = new HashSet<Vector2Int>();
    private GameObject plane;
    private Vector2 planeSize = new Vector2(0, 0);
    
    public List<Agent> agents = new List<Agent>();


    void Awake()
    {
        // Initialize the plane size
        InitializePlaneSize();
        
        // Initialize the comfort map
        comfortMap = new float[textureResolution, textureResolution];
        changedPixels = new bool[textureResolution, textureResolution];
        
        // Initialize the trail colors array
        trailColors = new Color[textureResolution * textureResolution];
        for (int i = 0; i < trailColors.Length; i++)
        {
            trailColors[i] = comfortColor0; // Start with the "zero comfort" color
        }
        
        // Create the render texture for the trail
        trailTexture = new RenderTexture(textureResolution, textureResolution, 0, RenderTextureFormat.ARGB32);
        trailTexture.enableRandomWrite = true; // Required for compute shader
        trailTexture.Create();
        
        // Apply the render texture to the plane material
        if (planeMaterial != null)
        {
            planeMaterial.mainTexture = trailTexture;
        }
       

        // Initialize the compute buffer for trail positions
        InitializeTrailPositionBuffer();

        // Initialize compute shader resources if shader is assigned
        if (trailComputeShader != null)
        {
            InitializeComputeResources();
        }
        else
        {
            Debug.Log("Trail Compute Shader not assigned. Using CPU for trail updates.");
        }
    }

    private void InitializeTrailPositionBuffer()
    {
        // Always release the old buffer first to prevent leaks
        if (trailPositionBuffer != null)
        {
            trailPositionBuffer.Release();
            trailPositionBuffer = null; // Set to null after releasing
        }

        // Only create a new buffer if we have agents
        if (agents != null && agents.Count > 0)
        {
            // Create buffer with at least 1 element to avoid zero-sized buffer
            int bufferSize = Mathf.Max(1, agents.Count);
            trailPositionBuffer = new ComputeBuffer(bufferSize, sizeof(float) * 2);
        }
        else
        {
            // Create a minimal buffer with 1 element to avoid null reference
            trailPositionBuffer = new ComputeBuffer(1, sizeof(float) * 2);
            Debug.LogWarning("No agents to initialize trail position buffer. Created minimal buffer.");
        }
    }

    void FixedUpdate()
    {
        // Clear the stepped cells set at the beginning of each frame
        steppedCellsThisFrame.Clear();
        changedPixelsList.Clear();
        
        // Reset changed pixels array
        for (int x = 0; x < textureResolution; x++)
        {
            for (int y = 0; y < textureResolution; y++)
            {
                changedPixels[x, y] = false;
            }
        }
        
        // Get the goal reached distance
        float goalReachedDistance = agentSpawner != null ? agentSpawner.goalReachedDistance : goalReachedThreshold;
        
        // Update each agent
        for (int i = agents.Count - 1; i >= 0; i--)
        {
            Agent agent = agents[i];
            if (agent == null) continue;
            
            // Check if agent has reached its goal
            float distanceToGoal = Vector3.Distance(agent.transform.position, agent.goalPosition);
            if (distanceToGoal < goalReachedDistance)
            {
                // Let the AgentSpawner handle the agent reaching its goal
                agentSpawner.HandleAgentReachedGoal(agent);
                continue; // Skip to next agent since this one is being handled
            
            }
            
            // Update agent movement
            UpdateAgentMovement(agent);
            
            // Create trail for agent every frame
            CreateTrailForAgent(agent);
        }
        
        // Update the comfort map every frame
        UpdateComfortMap();
    }

    private void UpdateTrail(Agent agent, int agentIndex)
    {
        Vector2 currentPos = WorldToTextureCoord(agent.transform.position);
        
        // Get previous position using the agent reference, not index
        if (!agentPreviousPositions.ContainsKey(agent))
        {
            agentPreviousPositions[agent] = currentPos;
            return; // Skip first frame
        }
        
        Vector2 previousPos = agentPreviousPositions[agent];
        
        // Simple check: if position hasn't changed significantly in texture space, skip
        if (Vector2.Distance(currentPos, previousPos) < 0.5f)
        {
            return;
        }
        
        // Draw line from previous to current position
        int x0 = Mathf.RoundToInt(currentPos.x);
        int y0 = Mathf.RoundToInt(currentPos.y);
        int x1 = Mathf.RoundToInt(previousPos.x);
        int y1 = Mathf.RoundToInt(previousPos.y);
        
        DrawLineOnTrailMap(x0, y0, x1, y1);
        
        // Update previous position for the next frame
        agentPreviousPositions[agent] = currentPos;
    }

    private void UpdatePathTexture()
    {
        // Create a texture and copy the color data
        Texture2D tempTexture = new Texture2D(textureResolution, textureResolution, TextureFormat.RGBA32, false);
        tempTexture.SetPixels(trailColors);
        tempTexture.Apply();
        
        // Copy the texture to the render texture
        RenderTexture prevRT = RenderTexture.active;
        RenderTexture.active = trailTexture;
        Graphics.Blit(tempTexture, trailTexture);
        RenderTexture.active = prevRT;
        
        // Clean up
        Destroy(tempTexture);
        
        // Reset the force update flag
        forceTextureUpdate = false;
       
    }

    private Vector3 CalculateGoalForce(Agent agent)
    {
        // Get agent and goal positions
        Vector3 agentPos = agent.transform.position;
        Vector3 goalPos = agent.goalPosition;
        Vector3 goalForce = (goalPos - agentPos).normalized;
        return goalForce;
    }

    private Vector3 CalculateAvoidanceForce(Agent agent)
    {
        Vector3 avoidanceForce = Vector3.zero;
        
        // Calculate repulsion from other agents
        foreach (Agent otherAgent in agents)
        {
            if (otherAgent == agent || otherAgent == null) continue;
            
            Vector3 direction = agent.transform.position - otherAgent.transform.position;
            float distance = direction.magnitude;
            
            // Apply repulsion force if within radius
            if (distance < agentRepulsionRadius)
            {
                // Normalize direction
                direction = direction.normalized;
                
                // Calculate repulsion force (stronger as agents get closer)
                float repulsionStrength = agentRepulsionForce * (1.0f - distance / agentRepulsionRadius);
                avoidanceForce += direction * repulsionStrength;
            }
        }
        
        return avoidanceForce;
    }

    private Vector3 CalculatePathForce(Agent agent)
    {
        // Get the agent's position
        Vector3 agentPos = agent.transform.position;
        Vector3 trailForce = Vector3.zero;
        
        
        if (currentPathFollowingMethod == PathFollowingMethod.HelbingsMethod)
        {
            // Helbing's method: Sample the entire comfort map
            for (int x = 0; x < textureResolution; x++)
            {
                for (int y = 0; y < textureResolution; y++)
                {
                    float G = comfortMap[x, y];
                    if (G > 0.1f) // Only consider cells with significant comfort
                    {
                        // Convert texture coordinates to world position
                        Vector3 cellWorldPos = TextureCoordToWorld(new Vector2(x, y));
                        
                        // Calculate force contribution from this point
                        trailForce += CalculateForceContribution(agentPos, cellWorldPos, G, HelbingsDistanceFactor_sigma);
                    }
                }
            }
            
            // Scale the trail force by the path follow strength
            trailForce *= pathFollowStrength;
        }
        else // Vision-based method
        {
            // For vision-based method, we only sample points in the agent's field of view
            List<SamplePoint> samplePoints = GetSamplePointsForAgent(agent);
            if (samplePoints == null || samplePoints.Count == 0)
            {
                return Vector3.zero;
            }
            
            // Find all points with trails
            List<SamplePoint> trailPoints = samplePoints.FindAll(p => p.hasTrail);
            if (trailPoints.Count == 0)
            {
                return Vector3.zero; // No trail points found
            }
            
            // Calculate weighted sum of forces from all trail points
            foreach (var point in trailPoints)
            {
                // Calculate force contribution from this sample point
                trailForce += CalculateForceContribution(
                    agentPos, 
                    point.position, 
                    point.comfortValue, 
                    VisualDistanceFactor_sigma);
            }
            
            // Scale the trail force by the visual path follow strength
            trailForce *= visualPathFollowStrength;
        }
        
        return trailForce;
    }

    // Helper method to calculate force contribution from a single point
    private Vector3 CalculateForceContribution(Vector3 agentPos, Vector3 pointPos, float comfortValue, float sigma)
    {
        // Calculate direction and distance
        Vector3 direction = pointPos - agentPos;
        float distance = direction.magnitude;
        
        if (distance <= 0.001f) // Avoid division by zero
            return Vector3.zero;
        
        // Calculate force contribution based on equation (2)
        // f_i,trail = ∫ d²r (r - r_i)/|r - r_i| * exp(-|r - r_i|/σ) * G(r)/(2πσ²)
        float distanceFactor = Mathf.Exp(-distance / sigma) / (2 * Mathf.PI * sigma * sigma);
        Vector3 forceContribution = direction.normalized * comfortValue * distanceFactor;
        
        return forceContribution;
    }

    public List<SamplePoint> GetSamplePointsForAgent(Agent agent)
    {
        if (agent == null)
            return null;
        
        // Return the cached sample points if they exist
        if (agentSamplePoints.TryGetValue(agent, out List<SamplePoint> points))
        {
            return points;
        }
        
        // Generate new sample points if none exist
        return GenerateSamplePoints(agent);
    }

     // --- Cleanup agent data on removal ---
    public void RemoveAgent(Agent agent)
    {
        if (agent == null) return;
        
        // Remove from our list
        agents.Remove(agent);
        
        // Clean up any dictionaries that reference this agent
        agentLastTrailUpdateFrame.Remove(agent);
        agentSamplePoints.Remove(agent);
        agentPreviousPositions.Remove(agent);
    }
    // -----------------------------------
    // 2. WorldToTextureCoord method
    public Vector2 WorldToTextureCoord(Vector3 worldPos)
    {
        // Get the plane's position and scale
        Vector3 planeCenter = plane.transform.position;
        Vector3 planeScale = plane.transform.localScale;
        
        // Default Unity plane is 10x10 units in the XZ plane
        float planeWidth = planeScale.x * 10f;  // X dimension
        float planeLength = planeScale.z * 10f; // Z dimension
        
        // Calculate position relative to plane center (only use X and Z)
        float relativeX = worldPos.x - planeCenter.x;
        float relativeZ = worldPos.z - planeCenter.z;
        
        // Convert to normalized coordinates (0-1) with flipping
        // The flipping depends on how your plane is oriented in the scene
        float normalizedX = 1.0f - ((relativeX / planeWidth) + 0.5f); // Flip X
        float normalizedZ = 1.0f - ((relativeZ / planeLength) + 0.5f); // Flip Z
        
        // Map to texture coordinates
        int texX = Mathf.RoundToInt(normalizedX * (textureResolution - 1));
        int texY = Mathf.RoundToInt(normalizedZ * (textureResolution - 1));
        
        // Clamp to valid texture coordinates
        texX = Mathf.Clamp(texX, 0, textureResolution - 1);
        texY = Mathf.Clamp(texY, 0, textureResolution - 1);
        
        return new Vector2(texX, texY);
    }

    // 3. TextureCoordToWorld method (inverse of WorldToTextureCoord)
    public Vector3 TextureCoordToWorld(Vector2 texCoord)
    {
        // Get the plane's position and scale
        Vector3 planeCenter = plane.transform.position;
        Vector3 planeScale = plane.transform.localScale;
        
        // Default Unity plane is 10x10 units in the XZ plane
        float planeWidth = planeScale.x * 10f;  // X dimension
        float planeLength = planeScale.z * 10f; // Z dimension
        
        // Convert from texture coordinates (0 to textureResolution-1) to normalized coordinates (0 to 1)
        float normalizedX = texCoord.x / (textureResolution - 1);
        float normalizedZ = texCoord.y / (textureResolution - 1);
        
        // Flip coordinates (reverse of WorldToTextureCoord)
        normalizedX = 1.0f - normalizedX;
        normalizedZ = 1.0f - normalizedZ;
        
        // Convert to relative position (-0.5 to 0.5)
        float relativeX = (normalizedX - 0.5f) * planeWidth;
        float relativeZ = (normalizedZ - 0.5f) * planeLength;
        
        // Convert to world position
        Vector3 worldPos = planeCenter + new Vector3(relativeX, 0, relativeZ);
        
        return worldPos;
    }

    private void InitializeComputeResources()
    {
        int resolution = textureResolution;

        // Create RenderTexture
        if (trailTexture == null || trailTexture.width != resolution)
        {
            if (trailTexture != null) trailTexture.Release(); // Release old one
            trailTexture = new RenderTexture(resolution, resolution, 0, RenderTextureFormat.ARGBFloat); // Use float format for intensity/alpha
            trailTexture.enableRandomWrite = true;
            trailTexture.Create();
            Debug.Log("Compute Shader RenderTexture created.");

             // Optional: Clear the RenderTexture initially
             ClearRenderTexture(trailTexture);
        }


        // Create ComputeBuffer (size depends on agent count, might need resizing in RegisterAgents)
        int agentCount = agents.Count > 0 ? agents.Count : 1; // Need at least size 1 buffer? Check shader logic.
        if (trailPositionBuffer == null || trailPositionBuffer.count != agentCount)
        {
             if (trailPositionBuffer != null) trailPositionBuffer.Release();
             // Buffer stores Vector2 (float2) - position
             trailPositionBuffer = new ComputeBuffer(agentCount, sizeof(float) * 2);
             Debug.Log($"Compute Shader Position Buffer created/resized for {agentCount} agents.");
        }
    }

     // Helper to clear RenderTexture
    private void ClearRenderTexture(RenderTexture rt)
    {
        RenderTexture previousActive = RenderTexture.active;
        RenderTexture.active = rt;
        GL.Clear(true, true, Color.clear); // Clear to transparent black
        RenderTexture.active = previousActive;
        Debug.Log("RenderTexture cleared.");
    }


    void OnDestroy()
    {
        // Release compute resources
        if (trailTexture != null)
        {
            trailTexture.Release();
            trailTexture = null;
        }
        if (trailPositionBuffer != null)
        {
            trailPositionBuffer.Release();
            trailPositionBuffer = null;
        }
    }

    void OnDisable()
    {
         // Also release on disable to be safe
         OnDestroy();
    }

    // Add this method to update the compute shader
    private void UpdateTrailComputeShader()
    {
        if (trailComputeShader == null || trailPositionBuffer == null)
        {
            Debug.LogWarning("Cannot update trail compute shader: missing components.");
            return;
        }

        // Create a texture to hold the trail colors
        Texture2D colorTexture = new Texture2D(textureResolution, textureResolution, TextureFormat.RGBA32, false);
        colorTexture.SetPixels(trailColors);
        colorTexture.Apply();

        // Set up the compute shader
        int kernelHandle = trailComputeShader.FindKernel("CSMain");
        
        // Create a render texture for the compute shader output
        RenderTexture outputTexture = new RenderTexture(textureResolution, textureResolution, 0);
        outputTexture.enableRandomWrite = true;
        outputTexture.Create();
        
        trailComputeShader.SetTexture(kernelHandle, "Result", outputTexture);
        trailComputeShader.SetTexture(kernelHandle, "ColorMap", colorTexture);
        trailComputeShader.SetFloat("trailWidth", trailWidth);
        trailComputeShader.SetInt("textureResolution", textureResolution);
        
        // Prepare trail positions
        Vector2[] trailPositions = new Vector2[agents.Count];
        for (int i = 0; i < agents.Count; i++)
        {
            if (agents[i] != null)
            {
                trailPositions[i] = WorldToTextureCoord(agents[i].transform.position);
            }
            else
            {
                trailPositions[i] = Vector2.zero; // Default for null agents
            }
        }
        trailPositionBuffer.SetData(trailPositions);

        // Set the buffer
        trailComputeShader.SetBuffer(kernelHandle, "trailPositions", trailPositionBuffer);

        // Dispatch the compute shader
        trailComputeShader.Dispatch(kernelHandle, Mathf.CeilToInt(textureResolution / 8.0f), Mathf.CeilToInt(textureResolution / 8.0f), 1);

        // Clean up
        Destroy(colorTexture);
        Destroy(outputTexture);
    }

    private void UpdateComfortMapVisualization()
    {
        // Only update pixels that have changed
        foreach (Vector2Int pixel in changedPixelsList)
        {
            int x = pixel.x;
            int y = pixel.y;
            
            // Get the comfort value
            float G = comfortMap[x, y];
            
            // Map comfort value to color
            Color pixelColor;
            float normalizedG = G / maxComfortLevel;
            
            if (normalizedG <= 0.001f)
                pixelColor = comfortColor0;
            else if (normalizedG < 0.2f)
                pixelColor = Color.Lerp(comfortColor0, comfortColor1, normalizedG / 0.2f);
            else if (normalizedG < 0.4f)
                pixelColor = Color.Lerp(comfortColor1, comfortColor2, (normalizedG - 0.2f) / 0.2f);
            else if (normalizedG < 0.6f)
                pixelColor = Color.Lerp(comfortColor2, comfortColor3, (normalizedG - 0.4f) / 0.2f);
            else if (normalizedG < 0.8f)
                pixelColor = Color.Lerp(comfortColor3, comfortColor4, (normalizedG - 0.6f) / 0.2f);
            else if (normalizedG < 0.9f)
                pixelColor = Color.Lerp(comfortColor4, comfortColor5, (normalizedG - 0.8f) / 0.1f);
            else
                pixelColor = Color.Lerp(comfortColor5, comfortColor6, (normalizedG - 0.9f) / 0.1f);
            
            // Update the color in the array
            int index = y * textureResolution + x;
            if (index >= 0 && index < trailColors.Length)
            {
                trailColors[index] = pixelColor;
            }
        }
        
        // Reset the changed pixels list after updating
        changedPixels = new bool[textureResolution, textureResolution];
    }

    private void UpdateComfortMap()
    {
        float deltaTime = Time.fixedDeltaTime;
        
        // Process stepped cells first (usually much fewer than the entire grid)
        foreach (Vector2Int cell in steppedCellsThisFrame)
        {
            int x = cell.x;
            int y = cell.y;
            
            if (x < 0 || x >= textureResolution || y < 0 || y >= textureResolution)
                continue;
            
            float G = comfortMap[x, y];
            
            // Implement equation (5) from the paper:
            // ∂G(~r, t)/∂t = -G(~r, t)/T(~r) + I(~r)(1 - G(~r, t)/Gmax(~r))∑δ(~r - ~ri)
            
            // Calculate saturation factor: (1 - G/Gmax)
            float saturationFactor = (maxComfortLevel > 0.001f) ? (1.0f - G / maxComfortLevel) : 0.0f;
            saturationFactor = Mathf.Max(0f, saturationFactor); // Ensure non-negative
            
            // Calculate comfort increase from footsteps: I * saturationFactor
            // The sum of delta functions is implicitly 1 here since we're only processing cells with footsteps
            float comfortIncrease = footstepIntensity_I * saturationFactor;
            
            // Update comfort value
            float newG = G + comfortIncrease;
            newG = Mathf.Min(newG, maxComfortLevel); // Cap at maxComfortLevel
            
            if (Mathf.Abs(newG - G) > 0.001f)
            {
                comfortMap[x, y] = newG;
                // Mark as changed for visualization update
                if (!changedPixels[x, y])
                {
                    changedPixels[x, y] = true;
                    changedPixelsList.Add(new Vector2Int(x, y));
                }
            }
        }
        
        // Process decay for ALL cells every frame
        for (int x = 0; x < textureResolution; x++)
        {
            for (int y = 0; y < textureResolution; y++)
            {
                float G = comfortMap[x, y];
                if (G > 0.001f)
                {
                    // Calculate decay: -G/T
                    float decayAmount = (G / trailRecoveryRate_T) * deltaTime;
                    
                    // Update comfort value
                    float newG = Mathf.Max(0f, G - decayAmount);
                    
                    if (Mathf.Abs(newG - G) > 0.001f)
                    {
                        comfortMap[x, y] = newG;
                        // Mark as changed for visualization update
                        if (!changedPixels[x, y])
                        {
                            changedPixels[x, y] = true;
                            changedPixelsList.Add(new Vector2Int(x, y));
                        }
                    }
                }
            }
        }
        
        // Update visualization every frame
        if (changedPixelsList.Count > 0)
        {
            forceTextureUpdate = true;
            UpdateComfortMapVisualization();
            UpdatePathTexture();
        }
    }

    private Vector2 GetPreviousPosition(Agent agent)
    {
        // Use a dictionary to store previous positions instead of a list
        if (!agentPreviousPositions.ContainsKey(agent))
        {
            agentPreviousPositions[agent] = WorldToTextureCoord(agent.transform.position);
        }
        return agentPreviousPositions[agent];
    }

    private void SetPreviousPosition(Agent agent, Vector2 position)
    {
        agentPreviousPositions[agent] = position;
    }

    private List<SamplePoint> GenerateSamplePoints(Agent agent)
    {
        if (agent == null) return new List<SamplePoint>();
        
        List<SamplePoint> points = new List<SamplePoint>();
        
        // Get agent's position and direction
        Vector3 agentPos = agent.transform.position;
        Vector3 forward = agent.currentDirection.normalized;
        
        // Ensure the direction is in the XZ plane
        forward.y = 0;
        if (forward.magnitude < 0.01f) forward = Vector3.forward;
        forward.Normalize();
        
        // Calculate right vector (perpendicular to forward in XZ plane)
        Vector3 right = Vector3.Cross(Vector3.up, forward).normalized;
        
        // Vision cone parameters
        float visionLength = agent.visionLength;
        float halfFOV = agent.fieldOfView / 2f * Mathf.Deg2Rad;
        
        // Get the plane's Y position
        float planeY = plane.transform.position.y;
        
        // Generate sample points in a grid within the vision cone
        int ringsCount = 5; // Number of distance rings
        int pointsPerRing = 7; // Number of points per ring
        
        for (int r = 1; r <= ringsCount; r++)
        {
            float distance = (visionLength * r) / ringsCount;
            
            for (int p = 0; p < pointsPerRing; p++)
            {
                // Calculate angle within FOV
                float angle = -halfFOV + (p * (2 * halfFOV) / (pointsPerRing - 1));
                
                // Calculate direction using rotation in XZ plane
                Vector3 direction = forward * Mathf.Cos(angle) + right * Mathf.Sin(angle);
                
                // Calculate position (ensure it's on the plane)
                Vector3 position = agentPos + direction * distance;
                position.y = planeY; // Set Y to plane's Y position
                
                // Create sample point
                SamplePoint point = new SamplePoint
                {
                    position = position,
                    hasTrail = false,
                    comfortValue = 0f
                };
                
                // Check if this point has a trail
                Vector2 texCoord = WorldToTextureCoord(position);
                int texX = Mathf.RoundToInt(texCoord.x);
                int texY = Mathf.RoundToInt(texCoord.y);
                
                if (texX >= 0 && texX < textureResolution && texY >= 0 && texY < textureResolution)
                {
                    float comfort = comfortMap[texX, texY];
                    if (comfort > 0.1f) // Threshold for considering a trail
                    {
                        point.hasTrail = true;
                        point.comfortValue = comfort;
                    }
                }
                
                points.Add(point);
            }
        }
        return points;
    }

    private void UpdateAgentMovement(Agent agent)
    {
        float deltaTime = Time.fixedDeltaTime;
        
        // Calculate desired direction (unit vector e in equation 1)
        Vector3 goalDirection = (agent.goalPosition - agent.transform.position).normalized;
        
        // Calculate path force
        Vector3 pathForce = CalculatePathForce(agent);
        
        // Calculate avoidance force
        Vector3 avoidanceForce = CalculateAvoidanceForce(agent);
        
        // Combine forces to get the desired direction (equation 4)
        Vector3 combinedForce = goalDirection * goalForceStrength + pathForce + avoidanceForce;
        Vector3 desiredDirection = combinedForce.normalized;
        
        // Calculate desired velocity (v0 * e in equation 1)
        float desiredSpeed = agentMaxSpeed_v0; // v0 in equation 1
        Vector3 desiredVelocity = desiredDirection * desiredSpeed;
        
        // Current velocity
        Vector3 currentVelocity = agent.currentDirection * agent.velocity;
        
        // Calculate acceleration using social force model (equation 1)
        // f = (1/τ)(v0*e - v)
        Vector3 socialForce = (desiredVelocity - currentVelocity) / relaxationTime_tau;
        
        // Store forces for visualization
        agent.lastGoalForce = goalDirection * goalForceStrength;
        agent.lastPathForce = pathForce;
        agent.lastAvoidanceForce = avoidanceForce;
        agent.lastTotalForce = socialForce;
        
        // Apply Verlet velocity algorithm (equations 6 and 7)
        Vector3 currentPos = agent.transform.position;
        
        // If this is the first update, initialize previousPosition
        if (agent.previousPosition == Vector3.zero)
        {
            agent.previousPosition = currentPos - currentVelocity * deltaTime;
        }
        
        // Update position using Verlet integration (equation 6)
        // r(t+Δ) = 2r(t) - r(t-Δ) + Δ²f(t)
        Vector3 newPos = 2 * currentPos - agent.previousPosition + socialForce * deltaTime * deltaTime;
        
        // Update velocity (equation 7)
        // v(t+Δ) = (r(t+Δ) - r(t))/Δ
        Vector3 newVelocity = (newPos - currentPos) / deltaTime;
        
        // Limit speed
        if (newVelocity.magnitude > agentMaxSpeed_v0)
        {
            newVelocity = newVelocity.normalized * agentMaxSpeed_v0;
            
            // Adjust position to match the capped velocity
            newPos = currentPos + newVelocity * deltaTime;
        }
        
        // Update agent position
        agent.transform.position = newPos;
        
        // Update agent direction and velocity
        if (newVelocity.magnitude > 0.01f)
        {
            agent.currentDirection = newVelocity.normalized;
            agent.velocity = newVelocity.magnitude;
        }
        
        // Store current position for next Verlet step
        agent.previousPosition = currentPos;
        
        // Keep agent at fixed Y position if specified
        if (agent.fixedY != 0)
        {
            Vector3 pos = agent.transform.position;
            pos.y = agent.fixedY;
            agent.transform.position = pos;
        }
    }

    private void CreateTrailForAgent(Agent agent)
    {
        
        // Get the agent's position but project it onto the plane (use the plane's Y position)
        Vector3 projectedPosition = agent.transform.position;
        projectedPosition.y = plane.transform.position.y; // Use the plane's Y position
        
        Vector2 texCoord = WorldToTextureCoord(projectedPosition);
        int texX = Mathf.RoundToInt(texCoord.x);
        int texY = Mathf.RoundToInt(texCoord.y);

        if (texX < 0 || texX >= textureResolution || texY < 0 || texY >= textureResolution)
        {
            agentPreviousPositions[agent] = texCoord;
            return;
        }

        // Register the step at the current position
        RegisterStepAt(texX, texY);

        if (agentPreviousPositions.TryGetValue(agent, out Vector2 prevTexCoord))
        {
            int prevTexX = Mathf.RoundToInt(prevTexCoord.x);
            int prevTexY = Mathf.RoundToInt(prevTexCoord.y);

            if (prevTexX >= 0 && prevTexX < textureResolution &&
                prevTexY >= 0 && prevTexY < textureResolution)
            {
                float distance = Vector2.Distance(texCoord, prevTexCoord);
                
                if (distance > 0.5f)
                {
                    // Draw line registers all steps along the path
                    DrawLineOnTrailMap(prevTexX, prevTexY, texX, texY);
                }
            }
        }
        
        agentPreviousPositions[agent] = texCoord;
    }

    private void RegisterStepAt(int centerX, int centerY)
    {
        // Calculate the radius based on trailWidth
        int radius = Mathf.FloorToInt(trailWidth / 2f);

        // Iterate in a square area around the center point
        for (int x = centerX - radius; x <= centerX + radius; x++)
        {
            for (int y = centerY - radius; y <= centerY + radius; y++)
            {
                // Check if coordinates are valid
                if (x >= 0 && x < textureResolution && y >= 0 && y < textureResolution)
                {
                    // Add to the set of cells stepped on this frame
                    steppedCellsThisFrame.Add(new Vector2Int(x, y));
                   
                }
            }
        }
    }

    private void DrawLineOnTrailMap(int x0, int y0, int x1, int y1)
    {
        // Bresenham's line algorithm
        int dx = Mathf.Abs(x1 - x0);
        int dy = Mathf.Abs(y1 - y0);
        int sx = x0 < x1 ? 1 : -1;
        int sy = y0 < y1 ? 1 : -1;
        int err = dx - dy;

        while (true)
        {
            // Register the step using the method that handles trail width
            RegisterStepAt(x0, y0); // Marks cells in the steppedCellsThisFrame set

            if (x0 == x1 && y0 == y1) break;
            int e2 = 2 * err;
            if (e2 > -dy)
            {
                err -= dy;
                x0 += sx;
            }
            if (e2 < dx)
            {
                err += dx;
                y0 += sy;
            }
        }
    }

    // Add this method to initialize agent velocities
    private void InitializeAgentVelocity(Agent agent)
    {
        // Set initial velocity based on the direction to the goal
        Vector3 goalDirection = (agent.goalPosition - agent.transform.position).normalized;
        agent.currentDirection = goalDirection;
        agent.velocity = agentMaxSpeed_v0 * 0.5f; // Start at half max speed
        
        // Initialize previous position for Verlet integration
        agent.previousPosition = agent.transform.position - agent.currentDirection * agent.velocity * Time.fixedDeltaTime;
    }

    // Call this from RegisterAgents
    public void RegisterAgents(List<Agent> newAgents)
    {
        if (newAgents == null || newAgents.Count == 0)
            return;
        
        // Add all new agents to our list
        foreach (Agent agent in newAgents)
        {
            if (agent != null && !agents.Contains(agent))
            {
                agents.Add(agent);
                
                // Initialize agent's previous position for trail creation
                agentPreviousPositions[agent] = WorldToTextureCoord(agent.transform.position);
                
                // Initialize agent's last trail update frame
                agentLastTrailUpdateFrame[agent] = Time.frameCount;
                
                // Initialize agent's velocity
                InitializeAgentVelocity(agent);
            }
        }
        
        // Reinitialize the trail position buffer to accommodate the new agents
        InitializeTrailPositionBuffer();
    }

    // Add this method to the Simulator class
    public void StartSimulation()
    {
        // This method is called by ScriptManager to start the simulation
        Debug.Log("Simulation started with " + agents.Count + " agents");
        
        // Initialize any simulation parameters if needed
        
        // Make sure the trail position buffer is initialized
        if (trailPositionBuffer == null || trailPositionBuffer.count != agents.Count)
        {
            InitializeTrailPositionBuffer();
        }
        
        // Force an initial update of the comfort map visualization
        forceTextureUpdate = true;
        UpdateComfortMapVisualization();
        UpdatePathTexture();
    }
    // Add this method to initialize the plane size
    private void InitializePlaneSize()
    {
        // Find the plane GameObject if not already set
        if (plane == null)
        {
            plane = GameObject.Find("Ground"); // Adjust name if needed
            if (plane == null)
            {
                Debug.LogError("Could not find the plane GameObject!");
                return;
            }
        }
        
        // Get the plane's scale
        Vector3 planeScale = plane.transform.localScale;
        
        // Get the plane's mesh size (assuming it's a default Unity plane which is 10x10 units)
        float defaultPlaneSize = 10f;
        
        // Calculate the actual plane size in world units
        planeSize = new Vector2(planeScale.x * defaultPlaneSize, planeScale.z * defaultPlaneSize);
    }

    // Completely revamp this method to ensure no trail is created
    public void ResetAgentTrailTracking(Agent agent, Vector3 newPosition)
    {
        if (agent == null) return;
        
        // First, remove the agent from all tracking dictionaries
        agentPreviousPositions.Remove(agent);
        agentLastTrailUpdateFrame.Remove(agent);
        agentSamplePoints.Remove(agent);
        
        // Then re-add it with the new position
        Vector2 newTexCoord = WorldToTextureCoord(newPosition);
        agentPreviousPositions.Add(agent, newTexCoord);
        agentLastTrailUpdateFrame.Add(agent, Time.frameCount);
        agentSamplePoints.Add(agent, new List<SamplePoint>());
        
        // Force the agent's previous position to be the same as current position
        // to prevent any velocity-based trail creation
        agent.previousPosition = newPosition;
        
        // Set the skip flag for the next frame
        //skipTrailCreation[agent] = true;
       
    }

    // Add this method to register a single agent
    public void RegisterSingleAgent(Agent agent)
    {
        if (agent == null || agents.Contains(agent))
            return;
        
        agents.Add(agent);
        
        // Initialize agent's previous position for trail creation
        agentPreviousPositions[agent] = WorldToTextureCoord(agent.transform.position);
        
        // Initialize agent's last trail update frame
        agentLastTrailUpdateFrame[agent] = Time.frameCount;
        
        // Initialize agent's velocity
        InitializeAgentVelocity(agent);
    }
}




