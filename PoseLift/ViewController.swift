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

class ViewController: UIViewController, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
    
    @IBOutlet weak var videoView: UIView!
    
    // video
    var videoURL: URL?
    var player: AVPlayer!
    var avpController = AVPlayerViewController()
    
    // gallery
    let galleryPicker = UIImagePickerController()

    override func viewDidLoad() {
        super.viewDidLoad()
        
    }
    
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

