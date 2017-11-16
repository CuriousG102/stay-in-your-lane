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
    private int i = 1;
    // private bool gotInstructions = false;

    // Use this for initialization
    void Start()
    {
        _socket = GameObject.Find("SocketIO").GetComponent<SocketIOComponent>();
        _carController = CarRemoteControl.GetComponent<CarController>();
        _socket.On("instructions", OnInstructions);
    }

    void OnInstructions(SocketIOEvent e) {
        JSONObject jsonObject = e.data;
        CarRemoteControl.SteeringAngle = float.Parse(jsonObject.GetField("steering_angle").str);
        CarRemoteControl.Acceleration = float.Parse(jsonObject.GetField("throttle").str);
        Time.timeScale = 1;
    }

    void FixedUpdate()
    {
        if (i < 100) {
            ++i;
            return;
        }
        Dictionary<string, string> data = new Dictionary<string, string>();
        data["steering_angle"] = _carController.CurrentSteerAngle.ToString("N4");
        data["throttle"] = _carController.AccelInput.ToString("N4");
        data["speed"] = _carController.CurrentSpeed.ToString("N4");
        data["front_image"] = Convert.ToBase64String(CameraHelper.CaptureFrame(FrontFacingCamera));
        data["overhead_image"] = Convert.ToBase64String(CameraHelper.CaptureFrame(OverheadCamera));
        data["delta_time"] = Time.deltaTime.ToString("N4");
        _socket.Emit("telemetry", new JSONObject(data));
        _socket.Emit("instruction", new JSONObject(new Dictionary<string, string>()));
        Time.timeScale = 0;
    }
}