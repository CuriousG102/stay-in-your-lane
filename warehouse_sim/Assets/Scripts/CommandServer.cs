using UnityEngine;
using System.Collections.Generic;
using System.Collections;
using SocketIO;
using UnityEngine.SceneManagement;
using UnityStandardAssets.Vehicles.Car;
using System;
using System.Security.AccessControl;

public class CommandServer : MonoBehaviour
{
    public CarRemoteControl CarRemoteControl;
    public CarController CarController;
    public Camera FrontFacingCamera;
    public Camera OverheadCamera;
    public Camera CheaterCamera;
    public bool ManualControl;
    private SocketIOComponent _socket;
    private CarController _carController;
    private int i = 1;
    // private bool gotInstructions = false;

    // Use this for initialization
    void Start()
    {
        _socket = GameObject.Find("SocketIO").GetComponent<SocketIOComponent>();
        _carController = CarController;
        if (!ManualControl) {
            _socket.On("instructions", OnInstructions);
        }
        Application.runInBackground = true;
    }

    void OnInstructions(SocketIOEvent e) {
        JSONObject jsonObject = e.data;
        string resetInstructions = jsonObject.GetField("reset").str;
        if (resetInstructions == "yes") {
            SceneManager.LoadScene(SceneManager.GetSceneAt(0).name);
            Time.timeScale = 1;
            return;
        }
        CarRemoteControl.SteeringAngle = float.Parse(jsonObject.GetField("steering_angle").str);
        CarRemoteControl.Acceleration = float.Parse(jsonObject.GetField("throttle").str);
        Time.timeScale = 1;
    }

    void EmitTelemetry() {
        Dictionary<string, string> data = new Dictionary<string, string>();
        data["steering_angle"] = _carController.CurrentSteerAngle.ToString("N4");
        data["throttle"] = _carController.AccelInput.ToString("N4");
        data["speed"] = (_carController.CurrentSpeed / 2.23693629f).ToString("N4") ;
        data["front_image"] = Convert.ToBase64String(CameraHelper.CaptureFrame(FrontFacingCamera));
        data["overhead_image"] = Convert.ToBase64String(CameraHelper.CaptureFrame(OverheadCamera));
        data["cheater_image"] = Convert.ToBase64String(CameraHelper.CaptureFrame(CheaterCamera));
        data["delta_time"] = Time.deltaTime.ToString("N4");
        Vector3 position = _carController.Position;
        data["x"] = position.x.ToString("N4");
        data["y"] = position.y.ToString("N4");
        data["z"] = position.z.ToString("N4");
        Quaternion rotation = _carController.Rotation;
        float angle;
        Vector3 angleAxis;
        rotation.ToAngleAxis(out angle, out angleAxis);
        data["rot_x"] = (angle * angleAxis.x).ToString("N4");
        data["rot_y"] = (angle * angleAxis.y).ToString("N4");
        data["rot_z"] = (angle * angleAxis.z).ToString("N4");
        data["is_colliding"] = _carController.IsColliding.ToString();
        data["is_finished"] = _carController.IsFinished.ToString();
        _socket.Emit("telemetry", new JSONObject(data));
    }

    void Update() {
        if (ManualControl) {
            EmitTelemetry();
        }
    }

    void FixedUpdate()
    {
        if (ManualControl) {
            return;
        }
        if (i < 100) {
            ++i;
            return;
        }
        EmitTelemetry();
        _socket.Emit("instruction", new JSONObject(new Dictionary<string, string>()));
        Time.timeScale = 0;
    }
}