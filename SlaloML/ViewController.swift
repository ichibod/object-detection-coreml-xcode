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
    
    var nmsThreshold: Float = 0.5
    
    var inputImage: CIImage?

    override func viewDidLoad() {
        super.viewDidLoad()
        guard let image = UIImage(contentsOfFile: Bundle.main.path(forResource: "20171211_224011963_iOS", ofType: "jpg")!) else {
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
    
    struct Prediction {
        let labelIndex: Int
        let confidence: Float
        let boundingBox: CGRect
    }
    
    func handleDetection(request: VNRequest, error: Error?) {
        let mlmodel = SlalomLogoClassifier()
        let userDefined: [String: String] = mlmodel.model.modelDescription.metadata[MLModelMetadataKey.creatorDefinedKey]! as! [String : String]
        nmsThreshold = Float(userDefined["non_maximum_suppression_threshold"]!) ?? 0.5
        
        guard let observations = request.results as? [VNCoreMLFeatureValueObservation]
            else { fatalError("unexpected result type from VNCoreMLRequest") }
        let predictions = ViewController.predictionsFromMultiDimensionalArrays(observations: observations, nmsThreshold: nmsThreshold)
        
        var strings: [String] = []
        if let predictions = predictions {
            for prediction in predictions {
                let pct = Float(Int(prediction.confidence * 10000)) / 100
                strings.append("\(pct)%")
                drawRectangle(detectedRectangle: prediction.boundingBox)
                
            }
        }
        DispatchQueue.main.async {
            self.classificationLabel.text = strings.joined(separator: ", ")
        }
    }

    // static only because the CameraViewController uses it too
    static func predictionsFromMultiDimensionalArrays(observations: [VNCoreMLFeatureValueObservation]?, nmsThreshold: Float = 0.5) -> [Prediction]? {
        guard let results = observations else {
            return nil
        }
        
        let coordinates = results[0].featureValue.multiArrayValue!
        let confidence = results[1].featureValue.multiArrayValue!
        
        let confidenceThreshold = 0.25
        var unorderedPredictions = [Prediction]()
        let numBoundingBoxes = confidence.shape[0].intValue
        let numClasses = confidence.shape[1].intValue
        let confidencePointer = UnsafeMutablePointer<Double>(OpaquePointer(confidence.dataPointer))
        let coordinatesPointer = UnsafeMutablePointer<Double>(OpaquePointer(coordinates.dataPointer))
        for b in 0..<numBoundingBoxes {
            var maxConfidence = 0.0
            var maxIndex = 0
            for c in 0..<numClasses {
                let conf = confidencePointer[b * numClasses + c]
                if conf > maxConfidence {
                    maxConfidence = conf
                    maxIndex = c
                }
            }
            if maxConfidence > confidenceThreshold {
                let x = coordinatesPointer[b * 4]
                let y = coordinatesPointer[b * 4 + 1]
                let w = coordinatesPointer[b * 4 + 2]
                let h = coordinatesPointer[b * 4 + 3]
                
                let rect = CGRect(x: CGFloat(x - w/2), y: CGFloat(y - h/2),
                                  width: CGFloat(w), height: CGFloat(h))
                
                let prediction = Prediction(labelIndex: maxIndex,
                                            confidence: Float(maxConfidence),
                                            boundingBox: rect)
                unorderedPredictions.append(prediction)
            }
        }
        
        var predictions: [Prediction] = []
        let orderedPredictions = unorderedPredictions.sorted { $0.confidence > $1.confidence }
        var keep = [Bool](repeating: true, count: orderedPredictions.count)
        for i in 0..<orderedPredictions.count {
            if keep[i] {
                predictions.append(orderedPredictions[i])
                let bbox1 = orderedPredictions[i].boundingBox
                for j in (i+1)..<orderedPredictions.count {
                    if keep[j] {
                        let bbox2 = orderedPredictions[j].boundingBox
                        if IoU(bbox1, bbox2) > nmsThreshold {
                            keep[j] = false
                        }
                    }
                }
            }
        }
        
        return predictions
    }
    
    public func drawRectangle(detectedRectangle: CGRect) {
        guard let inputImage = inputImage else {
            return
        }
        // Verify detected rectangle is valid.
        let boundingBox = detectedRectangle.scaled(to: inputImage.extent.size)
        guard inputImage.extent.contains(boundingBox) else {
            print("invalid detected rectangle");
            return
        }
        
        // Show the pre-processed image
        DispatchQueue.main.async {
            self.imageView.image = self.drawOnImage(source: self.imageView.image!, boundingRect: detectedRectangle)
        }
    }
    
    static public func IoU(_ a: CGRect, _ b: CGRect) -> Float {
        let intersection = a.intersection(b)
        let union = a.union(b)
        return Float((intersection.width * intersection.height) / (union.width * union.height))
    }
    
    fileprivate func drawOnImage(source: UIImage,
                                 boundingRect: CGRect) -> UIImage {
        UIGraphicsBeginImageContextWithOptions(source.size, false, 1)
        let context = UIGraphicsGetCurrentContext()!
        context.translateBy(x: source.size.width, y: 0)
        context.scaleBy(x: -1.0, y: 1.0)
        context.setLineJoin(.round)
        context.setLineCap(.round)
        context.setShouldAntialias(true)
        context.setAllowsAntialiasing(true)
        
        let rectWidth = source.size.width * boundingRect.size.width
        let rectHeight = source.size.height * boundingRect.size.height
        
        //draw image
        let rect = CGRect(x: 0, y:0, width: source.size.width, height: source.size.height)
        context.draw(source.cgImage!, in: rect)
        
        
        //draw bound rect
        var fillColor = UIColor.green
        fillColor.setFill()
        context.addRect(CGRect(x: boundingRect.origin.x * source.size.width, y:boundingRect.origin.y * source.size.height, width: rectWidth, height: rectHeight))
        
        //draw overlay
        fillColor = UIColor.red
        fillColor.setStroke()
        context.setLineWidth(12.0)
        context.drawPath(using: CGPathDrawingMode.stroke)
        
        let coloredImg : UIImage = UIGraphicsGetImageFromCurrentImageContext()!
        UIGraphicsEndImageContext()
        return coloredImg
    }

}

