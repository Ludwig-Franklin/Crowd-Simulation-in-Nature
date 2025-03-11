using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ScriptManager : MonoBehaviour
{
    private BasePathGenerator pathGen;
    private GradientGenerator gradGen;
    private BaseGoalManager goalManager;
    private AgentSpawner agentSpawner;
    private Simulator simulator;

    void Start()
    {
        pathGen = GetComponent<BasePathGenerator>();
        
        if (pathGen == null)
        {
            Debug.LogError("No BasePathGenerator component found! Please add either PathGenerator or CrossPathGenerator to this GameObject.");
            return;
        }
        
        gradGen = GetComponent<GradientGenerator>();
        agentSpawner = GetComponent<AgentSpawner>();
        goalManager = GetComponent<BaseGoalManager>();
        simulator = GetComponent<Simulator>();

        simulator.pathGen = pathGen;

        pathGen.CreateTexture();
        gradGen.GenerateGradient(pathGen.pathPixels);
        List<Agent> agents = agentSpawner.SpawnAgents();
        simulator.RegisterAgents(agents);
    }
}
