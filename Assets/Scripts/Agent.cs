using UnityEngine;

public class Agent : MonoBehaviour
{
    public float velocity;
    public float radius;
    public Vector3 goalPosition;
    // The agent's unique color.
    public Color agentColor = Color.white;

    public float fixedY;

    // Optionally, set the material color on awake.
    void Awake()
    {
        // Create a new instance of the material so that each agent can have its own color.
        Renderer rend = GetComponent<Renderer>();
        if (rend != null)
        {
            rend.material = new Material(rend.material);
            rend.material.color = agentColor;
        }
    }
}
