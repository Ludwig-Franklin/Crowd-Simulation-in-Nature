using UnityEngine;
using System.Collections.Generic;

public class PathGenerator : BasePathGenerator
{
    protected override void GeneratePaths()
    {
        List<List<Vector2>> allPaths = new List<List<Vector2>>();
        float res = textureResolution;
        float midX = res / 2f;

        // ***** MAIN PATH: oscillating from top to bottom *****
        List<Vector2> mainPath = new List<Vector2>()
        {
            new Vector2(midX, res - 1),                      // Top center
            new Vector2(midX + 50, 0.75f * res),
            new Vector2(midX - 50, 0.5f * res),
            new Vector2(midX + 50, 0.25f * res),
            new Vector2(midX, 0)                           // Bottom center
        };
        allPaths.Add(mainPath);

        // ***** LEFT SIDE: two branches from left edge merging and connecting to main *****
        Vector2 leftMeeting = new Vector2(res / 4f, 0.5f * res);
        List<Vector2> leftTop = new List<Vector2>()
        {
            new Vector2(0, 0.8f * res),
            new Vector2(res / 8f, 0.75f * res),
            leftMeeting
        };
        List<Vector2> leftBottom = new List<Vector2>()
        {
            new Vector2(0, 0.2f * res),
            new Vector2(res / 8f, 0.3f * res),
            leftMeeting
        };
        List<Vector2> leftConnection = new List<Vector2>()
        {
            leftMeeting,
            new Vector2((leftMeeting.x + midX) / 2f, 0.55f * res),
            new Vector2(midX, 0.5f * res)
        };
        allPaths.Add(leftTop);
        allPaths.Add(leftBottom);
        allPaths.Add(leftConnection);

        // ***** RIGHT SIDE: branches starting from near the middle (shifted right) *****
        Vector2 rightStartTop = new Vector2(midX + 15, 0.8f * res);
        Vector2 rightStartBottom = new Vector2(midX + 15, 0.2f * res);
        Vector2 rightMeeting = new Vector2((midX + (res - 1)) / 2f, 0.5f * res);
        List<Vector2> rightTop = new List<Vector2>()
        {
            rightStartTop,
            new Vector2((rightStartTop.x + rightMeeting.x)/2f, 0.75f * res),
            rightMeeting
        };
        List<Vector2> rightBottom = new List<Vector2>()
        {
            rightStartBottom,
            new Vector2((rightStartBottom.x + rightMeeting.x)/2f, 0.3f * res),
            rightMeeting
        };
        List<Vector2> rightExit = new List<Vector2>()
        {
            rightMeeting,
            new Vector2((rightMeeting.x + res - 1)/2f, 0.5f * res),
            new Vector2(res - 1, 0.5f * res)
        };
        allPaths.Add(rightTop);
        allPaths.Add(rightBottom);
        allPaths.Add(rightExit);

        // Use PaintPlane to draw all paths
        pathPixels = paintPlane.DrawPath(allPaths, pathColor, backgroundColor, pathWidth);
        paintPlane.ApplyTexture(plane);
    }
} 