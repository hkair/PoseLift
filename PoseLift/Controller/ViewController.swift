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
    
    public typealias DetectObjectsCompletion = ([PredictedPoint?]?, Error?) -> Void
    
    // MARK: - UI Properties
    @IBOutlet weak var videoView: UIView!
    @IBOutlet weak var jointView: DrawingJointView!
    @IBOutlet weak var labelsTableView: UITableView!
    
    // MARK: - Performance Measurement Property
    private let 👨‍🔧 = 📏()
    var isInferencing = false
    
    // MARK: - AV Property
    var videoURL: URL?
    var player: AVPlayer!
    var videoOutput: AVPlayerItemVideoOutput?
    
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
        
        // setup the Model
        setUpModel()
        
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

    
    // MARK: - predict Video
    func predictVideo() {
        player.currentItem?.addObserver(
            self,
            forKeyPath: #keyPath(AVPlayerItem.status),
            options: [.initial, .old, .new],
            context: nil)
        player.addPeriodicTimeObserver(
          forInterval: CMTime(value: 1, timescale: 30),
          queue: DispatchQueue(label: "videoProcessing", qos: .background),
          using: { time in
            self.predictUsingVision()
        })
    }
    
    override func observeValue(forKeyPath keyPath: String?, of object: Any?, change: [NSKeyValueChangeKey : Any]?, context: UnsafeMutableRawPointer?) {
      guard let keyPath = keyPath, let item = object as? AVPlayerItem
        else { return }

      switch keyPath {
      case #keyPath(AVPlayerItem.status):
        if item.status == .readyToPlay {
          self.setUpOutput()
        }
        break
      default: break
      }
    }
    
    func setUpOutput() {
      guard self.videoOutput == nil else { return }
      let videoItem = player.currentItem!
        if videoItem.status != AVPlayerItem.Status.readyToPlay {
        // see https://forums.developer.apple.com/thread/27589#128476
        return
      }

      let pixelBuffAttributes = [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
        ] as [String: Any]

      let videoOutput = AVPlayerItemVideoOutput(pixelBufferAttributes: pixelBuffAttributes)
      videoItem.add(videoOutput)
      self.videoOutput = videoOutput
    }

    func getNewFrame() -> CVPixelBuffer? {
      guard let videoOutput = videoOutput, let currentItem = player.currentItem else { return nil }

      let time = currentItem.currentTime()
      if !videoOutput.hasNewPixelBuffer(forItemTime: time) { return nil }
      guard let buffer = videoOutput.copyPixelBuffer(forItemTime: time, itemTimeForDisplay: nil)
        else { return nil }
      return buffer
    }
    
    // MARK: - SetUp Vide
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        if let URL = info[UIImagePickerController.InfoKey.mediaURL] as? URL {
            let player = AVPlayer(url: URL)
            let playerLayer = AVPlayerLayer(player: player)
            playerLayer.frame = self.view.bounds
            self.view.layer.addSublayer(playerLayer)
            self.player = player
            player.play()
            
            predictVideo()
            isInferencing = true
        }
        //playVideo()
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
    func predictUsingVision() {
        guard let pixelBuffer = getNewFrame() else { return }
        
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
        //self.👨‍🔧.🏷(with: "endInference")
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
            print(predictedPoints)
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
            //self.👨‍🔧.🎬🤚()
            self.isInferencing = false
            
            if #available(iOS 12.0, *) {
                os_signpost(.end, log: refreshLog, name: "PoseEstimation")
                }
            }
            /* =================================================================== */
            } else {
            // end of measure
            // from Measure Class
            //self.👨‍🔧.🎬🤚()
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
