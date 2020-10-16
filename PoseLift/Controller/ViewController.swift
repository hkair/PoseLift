//
//  ViewController.swift
//  PoseLift
//
//  Created by Hobin Kang on 2020-10-15.
//  Copyright © 2020 hobink. All rights reserved.
//

import UIKit
import AVKit
import AVFoundation

import CoreML
import Vision
import os.signpost

class ViewController: UIViewController, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
    
    let refreshLog = OSLog(subsystem: "com.hobink.PoseLift", category: "InferenceOperations")
    
    // MARK: - UI Properties
    @IBOutlet weak var videoView: UIView!
    @IBOutlet weak var jointView: DrawingJointView!
    @IBOutlet weak var labelsTableView: UITableView!
    
    // MARK: - Performance Measurement Property
    private let 👨‍🔧 = 📏()
    var isInferencing = false
    
    // MARK: - AV Property
    var videoURL: URL?
    var player: AVPlayer?
    var avpController = AVPlayerViewController()
    
    // MARK: - Gallery Property
    let galleryPicker = UIImagePickerController()
    
    // MARK: - ML Properties
    // Core ML model
    typealias EstimationModel = model_cpm // model name(model_cpm) must be equal with mlmodel file name
    
    // Preprocess and Inference
    var request: VNCoreMLRequest?
    var visionModel: VNCoreMLModel?
    
    // Postprocess
    var postProcessor: HeatmapPostProcessor = HeatmapPostProcessor()
    var mvfilters: [MovingAverageFilter] = []
    
    // Inference Result Data
    private var tableData: [PredictedPoint?] = []
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // setup tableview datasource on bottom
        labelsTableView.dataSource = self
        
        // setup delegate for performance measurement
        👨‍🔧.delegate = self
    }
    
    // MARK: - Setup Core ML
    func setUpModel() {
        if let visionModel = try? VNCoreMLModel(for: EstimationModel().model) {
            self.visionModel = visionModel
            request = VNCoreMLRequest(model: visionModel, completionHandler: visionRequestDidComplete)
            request?.imageCropAndScaleOption = .scaleFill
        } else {
            fatalError("cannot load the ml model")
        }
    }
    
    
    // MARK: - SetUp Vide
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        if let videoURL = info[UIImagePickerController.InfoKey.mediaURL] as? URL {
            player = AVPlayer(url: videoURL)
            avpController = AVPlayerViewController()
            avpController.player = player
            avpController.view.frame = videoView.frame
            self.addChild(avpController)
            self.view.addSubview(avpController.view)
        }
        dismiss(animated: true, completion: nil)
    }
    
    @IBAction func cameraTouched(_ sender: UIBarButtonItem) {
        galleryPicker.sourceType = .photoLibrary
        galleryPicker.delegate = self
        galleryPicker.mediaTypes = ["public.image", "public.movie"]
        present(galleryPicker, animated: true, completion: nil)
    }
    
}


extension ViewController {
    // MARK: - Inferencing
    func predictUsingVision(pixelBuffer: CVPixelBuffer) {
        guard let request = request else { fatalError() }
        // vision framework configures the input size of image following our model's input configuration automatically
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
        
        if #available(iOS 12.0, *) {
            os_signpost(.begin, log: refreshLog, name: "PoseEstimation")
        }
        try? handler.perform([request])
    }
    
    // MARK: - Postprocessing
    func visionRequestDidComplete(request: VNRequest, error: Error?) {
        if #available(iOS 12.0, *) {
            os_signpost(.event, log: refreshLog, name: "PoseEstimation")
        }
        self.👨‍🔧.🏷(with: "endInference")
        if let observations = request.results as? [VNCoreMLFeatureValueObservation],
            let heatmaps = observations.first?.featureValue.multiArrayValue {

            /* =================================================================== */
            /* ========================= post-processing ========================= */

            /* ------------------ convert heatmap to point array ----------------- */
            var predictedPoints = postProcessor.convertToPredictedPoints(from: heatmaps)

            /* --------------------- moving average filter ----------------------- */
            if predictedPoints.count != mvfilters.count {
                mvfilters = predictedPoints.map { _ in MovingAverageFilter(limit: 3) }
            }
            for (predictedPoint, filter) in zip(predictedPoints, mvfilters) {
                filter.add(element: predictedPoint)
            }
            predictedPoints = mvfilters.map { $0.averagedValue() }
            /* =================================================================== */

            /* =================================================================== */
            /* ======================= display the results ======================= */
            DispatchQueue.main.sync {
                // draw line
                self.jointView.bodyPoints = predictedPoints

                // show key points description
                self.showKeypointsDescription(with: predictedPoints)

                // end of measure
                // from Measure Class
                self.👨‍🔧.🎬🤚()
                self.isInferencing = false
                
                if #available(iOS 12.0, *) {
                    os_signpost(.end, log: refreshLog, name: "PoseEstimation")
                }
            }
            /* =================================================================== */
        } else {
            // end of measure
            // from Measure Class
            self.👨‍🔧.🎬🤚()
            self.isInferencing = false
            
            if #available(iOS 12.0, *) {
                os_signpost(.end, log: refreshLog, name: "PoseEstimation")
            }
        }
    }
    
    func showKeypointsDescription(with n_kpoints: [PredictedPoint?]) {
        self.tableData = n_kpoints
        self.labelsTableView.reloadData()
    }
}

// MARK: - UITableView Data Source
extension ViewController: UITableViewDataSource {
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return tableData.count// > 0 ? 1 : 0
    }
    
    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell: UITableViewCell = tableView.dequeueReusableCell(withIdentifier: "LabelCell", for: indexPath)
        cell.textLabel?.text = PoseEstimationForMobileConstant.pointLabels[indexPath.row]
        if let body_point = tableData[indexPath.row] {
            let pointText: String = "\(String(format: "%.3f", body_point.maxPoint.x)), \(String(format: "%.3f", body_point.maxPoint.y))"
            cell.detailTextLabel?.text = "(\(pointText)), [\(String(format: "%.3f", body_point.maxConfidence))]"
        } else {
            cell.detailTextLabel?.text = "N/A"
        }
        return cell
    }
}

// MARK: - 📏(Performance Measurement) Delegate
extension ViewController: 📏Delegate {
    func updateMeasure(inferenceTime: Double, executionTime: Double, fps: Int) {
        //self.inferenceLabel.text = "inference: \(Int(inferenceTime*1000.0)) ms"
        //self.etimeLabel.text = "execution: \(Int(executionTime*1000.0)) ms"
        //self.fpsLabel.text = "fps: \(fps)"
    }
}
