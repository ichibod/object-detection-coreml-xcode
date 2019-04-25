//
//  Copyright © 2017 Slalom. All rights reserved.
//

import UIKit
import CoreML
import Vision
import ImageIO

class ViewController: UIViewController {
    
    @IBOutlet var classificationLabel: UILabel!
    @IBOutlet var imageView: UIImageView!
    private var detectionOverlay: CALayer!
    
    var inputImage: CIImage?

    override func viewDidLoad() {
        super.viewDidLoad()
        guard let image = imageView.image else {
            return
        }
        guard let ciImage = CIImage(image: image)
            else { fatalError("can't create CIImage from UIImage") }
        
        let orientation = CGImagePropertyOrientation(image.imageOrientation)
        inputImage = ciImage.oriented(forExifOrientation: Int32(orientation.rawValue))
        
        // Show the image in the UI.
        imageView.image = image
        
        // Run the rectangle detector
        let handler = VNImageRequestHandler(ciImage: ciImage, orientation: orientation)
        DispatchQueue.global(qos: .userInteractive).async {
            do {
                try handler.perform([self.detectionRequest])
            } catch {
                print(error)
            }
        }
    }

    lazy var detectionRequest: VNCoreMLRequest = {
        // Load the ML model through its generated class and create a Vision request for it.
        do {
            let model = try VNCoreMLModel(for: SlalomLogoClassifier().model)
            
            return VNCoreMLRequest(model: model, completionHandler: self.handleDetection)
        } catch {
            fatalError("can't load Vision ML model: \(error)")
        }
    }()
    
    func handleDetection(request: VNRequest, error: Error?) {
        guard let results = request.results, !results.isEmpty else { return }

        var strings: [String] = []
        for observation in results where observation is VNRecognizedObjectObservation {
            guard let observation = observation as? VNRecognizedObjectObservation else {
                continue
            }
            let bestLabel = observation.labels.first! // Label with highest confidence
            let objectBounds = observation.boundingBox

            let pct = Float(Int(observation.confidence * 10000)) / 100
            strings.append("\(pct)%")
            drawRectangle(detectedRectangle: objectBounds, identifier: bestLabel.identifier, confidence: observation.confidence)

            print(bestLabel.identifier, observation.confidence, objectBounds)
        }

        DispatchQueue.main.async {
            self.classificationLabel.text = strings.joined(separator: ", ")
        }
    }
    
    public func drawRectangle(detectedRectangle: CGRect, identifier: String, confidence: VNConfidence) {
        DispatchQueue.main.async { [unowned self] in
            let boundingBox = VNImageRectForNormalizedRect(detectedRectangle, Int(self.imageView.frame.width), Int(self.imageView.frame.height))
            let layer = self.layerWithBounds(boundingBox, identifier: identifier, confidence: confidence)
            self.imageView.layer.addSublayer(layer)
        }
    }

    func layerWithBounds(_ rect: CGRect, identifier: String, confidence: VNConfidence) -> CALayer {
        let layer = CAShapeLayer()
        layer.bounds = rect
        layer.path = UIBezierPath(roundedRect: rect, cornerRadius: 8).cgPath
        layer.lineWidth = 3
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
        layer.position = CGPoint(x: imageView.frame.midX, y: imageView.frame.midY)
        return layer
    }
}

