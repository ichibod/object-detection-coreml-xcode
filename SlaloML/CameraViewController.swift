import UIKit
import AVFoundation
import Vision
import CoreML

class CameraViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    // Connect InterfaceBuilder views to code
    @IBOutlet weak var classificationText: UILabel!
    @IBOutlet weak var cameraView: UIView!
    
    private var requests = [VNRequest]()
    
    // Create a layer to display camera frames in the UIView
    private lazy var cameraLayer: AVCaptureVideoPreviewLayer = AVCaptureVideoPreviewLayer(session: self.captureSession)
    // Create an AVCaptureSession
    private lazy var captureSession: AVCaptureSession = {
        let session = AVCaptureSession()
        session.sessionPreset = AVCaptureSession.Preset.photo
        guard
            let backCamera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
            let input = try? AVCaptureDeviceInput(device: backCamera)
            else { return session }
        session.addInput(input)
        return session
    }()
    
    private lazy var classifier: SlalomLogoClassifier = SlalomLogoClassifier()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        cameraView?.layer.addSublayer(cameraLayer)
        cameraLayer.frame = cameraView.bounds
        
//        cameraLayer.videoGravity = .resizeAspectFill
        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)]
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "MyQueue"))
        self.captureSession.addOutput(videoOutput)
        self.captureSession.startRunning()
        setupVision()
    }
    
    func setupVision() {
        guard let visionModel = try? VNCoreMLModel(for: classifier.model) else {
            fatalError("Can’t load VisionML model")
        }
        let classificationRequest = VNCoreMLRequest(model: visionModel, completionHandler: handleClassifications)
        classificationRequest.imageCropAndScaleOption = VNImageCropAndScaleOption.scaleFill
        requests = [classificationRequest]
    }
    
    func handleClassifications(request: VNRequest, error: Error?) {
        
        let mlmodel = classifier
        let userDefined: [String: String] = mlmodel.model.modelDescription.metadata[MLModelMetadataKey.creatorDefinedKey]! as! [String : String]
        let nmsThreshold = Float(userDefined["non_maximum_suppression_threshold"]!) ?? 0.5
        
        guard let observations = request.results as? [VNCoreMLFeatureValueObservation] else {
            fatalError("unexpected result type from VNCoreMLRequest")
        }
        let predictions = ViewController.predictionsFromMultiDimensionalArrays(observations: observations, nmsThreshold: nmsThreshold)
        
        var strings: [String] = []
        if let predictions = predictions {
            for prediction in predictions {
                let pct = Float(Int(prediction.confidence * 10000)) / 100
                strings.append("\(pct)%")
            }
        }
        
        DispatchQueue.main.async {
            self.cameraLayer.sublayers?.removeSubrange(1...)
            
            if let predictions = predictions {
                for prediction in predictions {
                    self.highlightLogo(boundingRect: prediction.boundingBox)
                }
            }
            self.classificationText.text = strings.joined(separator: ", ")
        }
    }

    func highlightLogo(boundingRect: CGRect) {
        let source = self.cameraView.frame

        let rectWidth = source.size.width * boundingRect.size.width
        let rectHeight = source.size.height * boundingRect.size.height

        let outline = CALayer()
        outline.frame = CGRect(x: boundingRect.origin.x * source.size.width, y:boundingRect.origin.y * source.size.height, width: rectWidth, height: rectHeight)


        outline.borderWidth = 2.0
        outline.borderColor = UIColor.red.cgColor

        self.cameraLayer.addSublayer(outline)
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        self.cameraLayer.frame = self.cameraView?.bounds ?? .zero
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        var requestOptions:[VNImageOption : Any] = [:]
        if let cameraIntrinsicData = CMGetAttachment(sampleBuffer, kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix, nil) {
            requestOptions = [.cameraIntrinsics:cameraIntrinsicData]
        }
        let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: CGImagePropertyOrientation(rawValue: 6)!, options: requestOptions)
        do {
            try imageRequestHandler.perform(self.requests)
        } catch {
            print(error)
        }
    }
    
    @IBAction func dismiss(_ sender: Any) {
        captureSession.stopRunning()
        navigationController?.dismiss(animated: true, completion: nil)
    }
}
