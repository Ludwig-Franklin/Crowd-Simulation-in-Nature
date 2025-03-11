using UnityEngine;
using System.Collections.Generic;

public abstract class BasePathGenerator : MonoBehaviour
{
    [Header("Plane & Texture Settings")]
    public Vector2 planeSize = new Vector2(10f, 10f);
    public int textureResolution = 512;
    public Color backgroundColor = Color.black;
    public Color pathColor = Color.white;
    [Tooltip("Increase for a thicker trail.")]
    public float pathWidth = 4f;

    protected PaintPlane paintPlane;
    public HashSet<Vector2Int> pathPixels { get; protected set; }

    protected GameObject plane;

    protected virtual void Awake()
    {
        paintPlane = GetComponent<PaintPlane>();
        pathPixels = new HashSet<Vector2Int>();
    }

    public virtual void CreateTexture()
    {
        CreatePlane();
        paintPlane.Initialize(textureResolution);
        GeneratePaths();
    }

    protected virtual void CreatePlane()
    {
        plane = GameObject.CreatePrimitive(PrimitiveType.Plane);
        plane.transform.parent = transform;
        plane.transform.localPosition = Vector3.zero;
        plane.transform.localScale = new Vector3(planeSize.x / 10f, 1f, planeSize.y / 10f);
    }

    protected abstract void GeneratePaths();
} 