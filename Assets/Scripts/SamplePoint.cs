using UnityEngine;

public struct SamplePoint
{
    public Vector3 position;
    public bool hasTrail;
    public bool isChosenPoint;
    public float contributionWeight; // For weighted sum visualization
    public float comfortValue; // Add this field for comfort value
} 