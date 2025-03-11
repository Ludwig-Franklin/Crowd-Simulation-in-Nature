using UnityEngine;
using System.Collections.Generic;

public class AgentSpawner : MonoBehaviour
{
    public GameObject agentPrefab;
    public int numberOfAgents = 10;
    public float spawnMargin = 1f;
    [SerializeField] private float spawnHeight = 1.0f;

    // Pre-defined array of colors to assign (cycle through these for each agent)
    public Color[] agentColors = new Color[] { Color.red, Color.green, Color.blue, Color.yellow, Color.magenta, Color.cyan };

    private BasePathGenerator pathGen;
    private BaseGoalManager goalManager;
    private Simulator simulator;

    void Awake()
    {
        pathGen = GetComponent<BasePathGenerator>();
        goalManager = GetComponent<BaseGoalManager>();
    }

    public List<Agent> SpawnAgents()
    {
        List<Agent> agentList = new List<Agent>();
        Vector2 planeSize = pathGen.planeSize;

        for (int i = 0; i < numberOfAgents; i++)
        {
            Vector3 spawnPos = new Vector3(
                Random.Range(-planeSize.x / 2f + spawnMargin, planeSize.x / 2f - spawnMargin),
                spawnHeight,
                Random.Range(-planeSize.y / 2f + spawnMargin, planeSize.y / 2f - spawnMargin)
            );

            Agent agent = Instantiate(agentPrefab, spawnPos, Quaternion.identity).GetComponent<Agent>();
            // Set the agent's radius (assumes agent's prefab scale represents its size)
            agent.radius = agentPrefab.transform.localScale.x / 2;

            // Assign a unique color from the array (cycling through if there are more agents than colors)
            Color assignedColor = agentColors[i % agentColors.Length];
            agent.agentColor = assignedColor;
            agent.fixedY = spawnHeight;

            // Update the agent's material color (if the agent script's Start hasn't been called yet, you can also do this here)
            Renderer rend = agent.GetComponent<Renderer>();
            if (rend != null)
            {
                rend.material = new Material(rend.material);
                rend.material.color = assignedColor;
            }

            // Assign a new random goal for the agent.
            goalManager.AssignNewGoal(agent);
            agentList.Add(agent);
        }
        return agentList;
    }
}
