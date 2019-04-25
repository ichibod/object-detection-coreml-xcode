import UIKit
import AVFoundation
import Vision
import CoreML

enum Label: String {
    case buildSquare = "_b"
    case buildLogo = "_build"
    case slalomLogo = "Slalom"
}

final class CameraViewController: UIViewController {
    @IBOutlet private weak var classificationText: UILabel!
    @IBOutlet private weak var cameraView: UIView!
    private var bufferSize: CGSize = .zero

    private let videoDataOutputQueue = DispatchQueue(label: "VideoDataOutput", qos: .userInitiated, attributes: [], autoreleaseFrequency: .workItem)
    
    private var requests = [VNRequest]()
    
    // Create a layer to display camera frames in the UIView
    private var cameraLayer: AVCaptureVideoPreviewLayer!
    private var detectionOverlay: CALayer! = nil
    // Create an AVCaptureSession
    private let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    
    private lazy var classifier: SlalomLogoClassifier = SlalomLogoClassifier()
    
    override func viewDidLoad() {
        super.viewDidLoad()

        setUpAVCapture()
        setUpLayers()
        updateLayerGeometry()
        setUpVision()
        session.startRunning()
    }

    private func setUpAVCapture() {
        session.sessionPreset = AVCaptureSession.Preset.photo
        let backCamera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)
        let deviceInput: AVCaptureDeviceInput
        do {
            deviceInput = try AVCaptureDeviceInput(device: backCamera!)
        } catch {
            print("Could not create video device input: \(error)")
            return
        }
        session.beginConfiguration()
        session.sessionPreset = .vga640x480
        guard session.canAddInput(deviceInput) else {
            print("Could not add video device input to the session")
            session.commitConfiguration()
            return
        }

        if session.canAddOutput(videoOutput) {
            videoOutput.alwaysDiscardsLateVideoFrames = true
            videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)]
            videoOutput.setSampleBufferDelegate(self, queue: videoDataOutputQueue)
        }
        self.session.addOutput(videoOutput)
        let captureConnection = videoOutput.connection(with: .video)
        // Always process the frames
        captureConnection?.isEnabled = true
        do {
            try backCamera!.lockForConfiguration()
            let dimensions = CMVideoFormatDescriptionGetDimensions((backCamera?.activeFormat.formatDescription)!)
            bufferSize.width = CGFloat(dimensions.width)
            bufferSize.height = CGFloat(dimensions.height)
            backCamera!.unlockForConfiguration()
        } catch {
            print(error)
        }
        session.addInput(deviceInput)
        session.commitConfiguration()

        cameraLayer = AVCaptureVideoPreviewLayer(session: session)
        cameraLayer.videoGravity = .resizeAspectFill
        cameraLayer.frame = cameraView.bounds
        cameraView?.layer.addSublayer(cameraLayer)
    }

    private func setUpLayers() {
        detectionOverlay = CALayer() // container layer that has all the renderings of the observations
        detectionOverlay.name = "DetectionOverlay"
        detectionOverlay.backgroundColor = UIColor.orange.withAlphaComponent(0.2).cgColor
        detectionOverlay.bounds = CGRect(x: 0.0,
                                         y: 0.0,
                                         width: bufferSize.width,
                                         height: bufferSize.height)
        detectionOverlay.position = CGPoint(x: cameraView.layer.bounds.midX, y: cameraView.layer.bounds.midY)
        cameraView?.layer.addSublayer(detectionOverlay)
    }
    
    func setUpVision() {
        guard let visionModel = try? VNCoreMLModel(for: classifier.model) else {
            fatalError("Can’t load VisionML model")
        }
        let classificationRequest = VNCoreMLRequest(model: visionModel) { request, error in
            DispatchQueue.main.async { [weak self] in
                self?.handleClassifications(request: request, error: error)
            }
        }
        classificationRequest.imageCropAndScaleOption = VNImageCropAndScaleOption.scaleFill
        requests = [classificationRequest]
    }
    
    func handleClassifications(request: VNRequest, error: Error?) {
        detectionOverlay.removeAllAnimations()
        if let layers = detectionOverlay.sublayers {
            for layer in layers {
                layer.removeFromSuperlayer()
            }
        }
        guard let results = request.results, !results.isEmpty else { return }

        CATransaction.begin()
        CATransaction.setValue(kCFBooleanTrue, forKey: kCATransactionDisableActions)
        for observation in results where observation is VNRecognizedObjectObservation {
            guard let objectObservation = observation as? VNRecognizedObjectObservation else {
                continue
            }
            // Select only the label with the highest confidence.
            let topLabelObservation = objectObservation.labels[0]

            let objectBounds = VNImageRectForNormalizedRect(objectObservation.boundingBox, Int(bufferSize.width), Int(bufferSize.height))

            let shapeLayer = layerWithBounds(objectBounds, identifier: topLabelObservation.identifier, confidence: objectObservation.confidence)
            detectionOverlay.addSublayer(shapeLayer)
        }
        CATransaction.commit()
        self.updateLayerGeometry()
    }

    func layerWithBounds(_ rect: CGRect, identifier: String, confidence: VNConfidence) -> CALayer {
        let layer = CAShapeLayer()
        layer.bounds = rect
        layer.path = UIBezierPath(roundedRect: rect, cornerRadius: 8).cgPath
        layer.lineWidth = 8
        if confidence < 0.45 {
            layer.lineDashPattern = [8,12]
        }
        switch identifier {
        case Label.buildLogo.rawValue:
            layer.strokeColor = UIColor(named: "BuildCyan")!.cgColor
        case Label.buildSquare.rawValue:
            layer.strokeColor = UIColor(named: "BuildCyan")!.cgColor
        default:
            layer.strokeColor = UIColor.red.cgColor
        }
        layer.shadowOpacity = 0.7
        layer.shadowOffset = CGSize.zero
        layer.fillColor = UIColor.clear.cgColor
        layer.position = CGPoint(x: rect.midX, y: rect.midY)
        return layer
    }
    
    @IBAction func dismiss(_ sender: Any) {
        session.stopRunning()
        navigationController?.dismiss(animated: true, completion: nil)
    }

    public func exifOrientationFromDeviceOrientation() -> CGImagePropertyOrientation {
        let curDeviceOrientation = UIDevice.current.orientation
        let exifOrientation: CGImagePropertyOrientation

        switch curDeviceOrientation {
        case UIDeviceOrientation.portraitUpsideDown:  // Device oriented vertically, home button on the top
            exifOrientation = .left
        case UIDeviceOrientation.landscapeLeft:       // Device oriented horizontally, home button on the right
            exifOrientation = .upMirrored
        case UIDeviceOrientation.landscapeRight:      // Device oriented horizontally, home button on the left
            exifOrientation = .down
        case UIDeviceOrientation.portrait:            // Device oriented vertically, home button on the bottom
            exifOrientation = .up
        default:
            exifOrientation = .up
        }
        return exifOrientation
    }

    func updateLayerGeometry() {
        let bounds = cameraView.layer.bounds
        var scale: CGFloat

        let xScale: CGFloat = bounds.size.width / bufferSize.height
        let yScale: CGFloat = bounds.size.height / bufferSize.width

        scale = fmax(xScale, yScale)
        if scale.isInfinite {
            scale = 1.0
        }
        CATransaction.begin()
        CATransaction.setValue(kCFBooleanTrue, forKey: kCATransactionDisableActions)

        // rotate the layer into screen orientation and scale and mirror
        detectionOverlay.setAffineTransform(CGAffineTransform(rotationAngle: CGFloat(.pi / 2.0)).scaledBy(x: scale, y: -scale))
        // center the layer
        detectionOverlay.position = CGPoint (x: bounds.midX, y: bounds.midY)

        CATransaction.commit()

    }
}

extension CameraViewController: AVCaptureVideoDataOutputSampleBufferDelegate {

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        let orientation = exifOrientationFromDeviceOrientation()
        let imageRequestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: orientation, options: [:])
        do {
            try imageRequestHandler.perform(self.requests)
        } catch {
            print(error)
        }
    }

}
