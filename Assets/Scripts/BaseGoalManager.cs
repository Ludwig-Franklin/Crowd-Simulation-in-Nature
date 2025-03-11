using UnityEngine;
using System.Collections.Generic;
using System.Collections;

public abstract class BaseGoalManager : MonoBehaviour
{
    protected float goalHeight = 1.5f;
    public float goalReachedThreshold = 0.75f;

    // Dictionary to track each agent's assigned goal.
    protected Dictionary<Agent, Vector3> agentGoals = new Dictionary<Agent, Vector3>();
    protected BasePathGenerator pathGen;

    protected virtual void Awake()
    {
        pathGen = GetComponent<BasePathGenerator>();
    }

    // Abstract method to be implemented by derived classes
    public abstract void AssignNewGoal(Agent agent);

    // Checks if agents have reached their goals and assigns new ones if needed.
    public virtual void UpdateGoals()
    {
        List<Agent> agentsNeedingNewGoals = new List<Agent>();
        foreach (KeyValuePair<Agent, Vector3> kv in agentGoals)
        {
            Agent agent = kv.Key;
            if (Vector3.Distance(agent.transform.position, kv.Value) < goalReachedThreshold)
            {
                agentsNeedingNewGoals.Add(agent);
            }
        }
        foreach (Agent agent in agentsNeedingNewGoals)
        {
            AssignNewGoal(agent);
        }
    }

    protected virtual void Update()
    {
        UpdateGoals();
    }

    protected virtual void OnDrawGizmos()
    {
        if (agentGoals == null) return;

        // Draw agent goals
        foreach (var kv in agentGoals)
        {
            Gizmos.color = kv.Key.agentColor;
            Gizmos.DrawSphere(kv.Value, 0.2f);
        }
    }
} 