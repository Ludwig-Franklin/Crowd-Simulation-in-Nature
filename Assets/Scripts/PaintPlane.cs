using UnityEngine;
using System.Collections.Generic;

public class PaintPlane : MonoBehaviour
{
    private PathGenerator pathGen;
    private Texture2D texture;
    private int textureResolution;

    private void Awake()
    {
        pathGen = GetComponent<PathGenerator>();
    }
    public void Initialize(int resolution)
    {
        textureResolution = resolution;
        texture = new Texture2D(textureResolution, textureResolution);
    }

    public void PaintPixel(Vector2Int position, Color color)
    {
        texture.SetPixel(position.x, position.y, color);
    }

    public HashSet<Vector2Int> DrawPath(List<List<Vector2>> controlPointsList, Color pathColor, Color backgroundColor, float pathWidth = 4f)
    {
        // Clear texture with background color
        for (int x = 0; x < textureResolution; x++)
        {
            for (int y = 0; y < textureResolution; y++)
            {
                texture.SetPixel(x, y, backgroundColor);
            }
        }

        // Draw Bezier curves between keypoints
        HashSet<Vector2Int> pathPixels = new HashSet<Vector2Int>();
        foreach (var points in controlPointsList)
        {
            DrawBezierCurve(points, 400, pathColor, pathWidth, pathPixels);
        }
        
        texture.Apply();
        return pathPixels;
    }

    // Draw a Bezier curve by sampling many points and drawing smooth lines between them.
    private void DrawBezierCurve(List<Vector2> controlPoints, int sampleResolution, Color pathColor, float pathWidth, HashSet<Vector2Int> pathPixels)
    {
        if (controlPoints == null || controlPoints.Count < 2)
            return;
        Vector2 previousPoint = CalculateBezierPoint(0f, controlPoints);
        int steps = sampleResolution;
        for (int i = 1; i <= steps; i++)
        {
            float t = i / (float)steps;
            Vector2 currentPoint = CalculateBezierPoint(t, controlPoints);
            DrawLine(previousPoint, currentPoint, pathColor, pathWidth, pathPixels);
            previousPoint = currentPoint;
        }
    }

    // Calculate a point on an n-point Bezier curve using the Bernstein polynomial.
    private Vector2 CalculateBezierPoint(float t, List<Vector2> controlPoints)
    {
        int n = controlPoints.Count - 1;
        Vector2 point = Vector2.zero;
        for (int i = 0; i <= n; i++)
        {
            float binCoeff = BinomialCoefficient(n, i);
            float term = binCoeff * Mathf.Pow(1 - t, n - i) * Mathf.Pow(t, i);
            point += term * controlPoints[i];
        }
        return point;
    }

    private int BinomialCoefficient(int n, int k)
    {
        int result = 1;
        for (int i = 1; i <= k; i++)
        {
            result *= n--;
            result /= i;
        }
        return result;
    }

    // Draws a smooth line between two points by interpolating intermediate points.
    private void DrawLine(Vector2 start, Vector2 end, Color pathColor, float pathWidth, HashSet<Vector2Int> pathPixels)
    {
        float distance = Vector2.Distance(start, end);
        int lineSteps = Mathf.CeilToInt(distance);
        for (int i = 0; i <= lineSteps; i++)
        {
            float t = i / (float)lineSteps;
            Vector2 point = Vector2.Lerp(start, end, t);
            // Round to the nearest integer to accurately capture all pixels
            int x = Mathf.RoundToInt(point.x);
            int y = Mathf.RoundToInt(point.y);
            SetPixelWithWidth(x, y, pathColor, pathWidth, pathPixels);
        }
    }

    // Sets a pixel (with thickness) and adds its coordinate to the stored path pixels.
    private void SetPixelWithWidth(int x, int y, Color pathColor, float pathWidth, HashSet<Vector2Int> pathPixels)
    {
        int halfWidth = Mathf.CeilToInt(pathWidth / 2f);
        for (int i = -halfWidth; i <= halfWidth; i++)
        {
            for (int j = -halfWidth; j <= halfWidth; j++)
            {
                int px = Mathf.Clamp(x + i, 0, textureResolution - 1);
                int py = Mathf.Clamp(y + j, 0, textureResolution - 1);
                texture.SetPixel(px, py, pathColor);
                pathPixels.Add(new Vector2Int(px, py));
            }
        }
    }

    public void PaintGradient(List<List<float>> distanceMap)
    {
        // Find maximum distance for normalization
        float maxDistance = 0;
        for (int x = 0; x < textureResolution; x++)
        {
            for (int y = 0; y < textureResolution; y++)
            {
                maxDistance = Mathf.Max(maxDistance, distanceMap[x][y]);
            }
        }

        // Paint gradient
        Color tealColor = new Color(0, 0.5f, 0.5f, 1f); // Teal color
        for (int x = 0; x < textureResolution; x++)
        {
            for (int y = 0; y < textureResolution; y++)
            {
                float normalizedDistance = distanceMap[x][y] / maxDistance;
                Color pixelColor = Color.Lerp(Color.white, tealColor, normalizedDistance);
                texture.SetPixel(x, y, pixelColor);
            }
        }
        
        texture.Apply();
    }

    public Texture2D GetTexture()
    {
        return texture;
    }

    public void ApplyTexture(GameObject plane)
    {
        texture.Apply();
        Material material = new Material(Shader.Find("Standard"));
        material.mainTexture = texture;
        plane.GetComponent<Renderer>().material = material;
    }
} 