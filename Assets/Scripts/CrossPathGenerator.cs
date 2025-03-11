using UnityEngine;
using System.Collections.Generic;

public class CrossPathGenerator : BasePathGenerator
{
    protected override void GeneratePaths()
    {
        List<List<Vector2>> allPaths = new List<List<Vector2>>();
        float res = textureResolution;
        float midX = res / 2f;
        float midY = res / 2f;

        // ***** HORIZONTAL PATH: left to right *****
        List<Vector2> horizontalPath = new List<Vector2>()
        {
            new Vector2(0, midY),           // Left middle
            new Vector2(res - 1, midY)      // Right middle
        };
        allPaths.Add(horizontalPath);

        // ***** VERTICAL PATH: top to bottom *****
        List<Vector2> verticalPath = new List<Vector2>()
        {
            new Vector2(midX, 0),           // Bottom middle
            new Vector2(midX, res - 1)      // Top middle
        };
        allPaths.Add(verticalPath);

        // Use PaintPlane to draw all paths
        pathPixels = paintPlane.DrawPath(allPaths, pathColor, backgroundColor, pathWidth);
        paintPlane.ApplyTexture(plane);
    }
} 