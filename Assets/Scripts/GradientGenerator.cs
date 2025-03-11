using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GradientGenerator : MonoBehaviour
{
    private BasePathGenerator pathGen;
    private PaintPlane paintPlane;
    private Queue<Vector2Int> gradientQueue;
    public List<List<float>> distanceMap;
    public float maxDistance = 0;
    
    private void Awake()
    {
        pathGen = GetComponent<BasePathGenerator>();
        paintPlane = GetComponent<PaintPlane>();
        distanceMap = new List<List<float>>();
    }

    public void GenerateGradient(HashSet<Vector2Int> pixelLocations) 
    {
        // Initialize distance map with -1 (unvisited)
        distanceMap = new List<List<float>>();
        for (int i = 0; i < pathGen.textureResolution; i++)
        {
            List<float> innerList = new List<float>(pathGen.textureResolution);
            for (int j = 0; j < pathGen.textureResolution; j++)
            {
                innerList.Add(-1);
            }
            distanceMap.Add(innerList);
        }

        gradientQueue = new Queue<Vector2Int>();
        foreach (Vector2Int pixel in pixelLocations)
        {
            gradientQueue.Enqueue(pixel);
            distanceMap[pixel.x][pixel.y] = 0;
        }

        // Directions for adjacent pixels (up, right, down, left, and diagonals)
        Vector2Int[] directions = new Vector2Int[] {
            new Vector2Int(0, 1),    // up
            new Vector2Int(1, 0),    // right
            new Vector2Int(0, -1),   // down
            new Vector2Int(-1, 0),   // left
            new Vector2Int(1, 1),    // up-right
            new Vector2Int(1, -1),   // down-right
            new Vector2Int(-1, -1),  // down-left
            new Vector2Int(-1, 1)    // up-left
        };
        
        // Corresponding costs for each direction
        float[] costs = new float[] {
            1.0f,       // up
            1.0f,       // right
            1.0f,       // down
            1.0f,       // left
            1.414f,     // up-right (√2)
            1.414f,     // down-right (√2)
            1.414f,     // down-left (√2)
            1.414f      // up-left (√2)
        };

        // BFS to calculate distances
        while (gradientQueue.Count > 0) 
        {
            Vector2Int current = gradientQueue.Dequeue();
            float currentDistance = distanceMap[current.x][current.y];

            // Check all adjacent pixels
            for (int i = 0; i < directions.Length; i++)
            {
                Vector2Int dir = directions[i];
                float cost = costs[i];
                Vector2Int neighbor = current + dir;
                
                // Check if neighbor is within bounds
                if (neighbor.x >= 0 && neighbor.x < pathGen.textureResolution &&
                    neighbor.y >= 0 && neighbor.y < pathGen.textureResolution)
                {
                    float newDistance = currentDistance + cost;
                    
                    // If unvisited or we found a shorter path
                    if (distanceMap[neighbor.x][neighbor.y] == -1 || 
                        newDistance < distanceMap[neighbor.x][neighbor.y])
                    {
                        distanceMap[neighbor.x][neighbor.y] = newDistance;
                        gradientQueue.Enqueue(neighbor);
                        maxDistance = Mathf.Max(maxDistance, newDistance);
                    }
                }
            }
        }
        
        // Use PaintPlane to paint the gradient
        paintPlane.PaintGradient(distanceMap);
    }
}
