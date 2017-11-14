using UnityEngine;
using System.Collections.Generic;
using System.Collections;
using SocketIO;
using UnityStandardAssets.Vehicles.Car;
using System;
using System.Security.AccessControl;

public class CommandServer : MonoBehaviour
{
    public CarRemoteControl CarRemoteControl;
    public Camera FrontFacingCamera;
    public Camera OverheadCamera;
    private SocketIOComponent _socket;
    private CarController _carController;

    // Use this for initialization
    void Start()
    {
        _socket = GameObject.Find("SocketIO").GetComponent<SocketIOComponent>();
        _carController = CarRemoteControl.GetComponent<CarController>();
    }

    void Update()
    {
        Dictionary<string, string> data = new Dictionary<string, string>();
        data["steering_angle"] = _carController.CurrentSteerAngle.ToString("N4");
        data["throttle"] = _carController.AccelInput.ToString("N4");
        data["speed"] = _carController.CurrentSpeed.ToString("N4");
        data["front_image"] = Convert.ToBase64String(CameraHelper.CaptureFrame(FrontFacingCamera));
        data["overhead_image"] = Convert.ToBase64String(CameraHelper.CaptureFrame(OverheadCamera));
        _socket.Emit("telemetry", new JSONObject(data));
    }
}