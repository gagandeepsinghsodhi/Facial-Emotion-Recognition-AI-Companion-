"""
Module for real-time emotion analysis from video stream.
"""

import cv2
import numpy as np
import logging
import sys
import signal
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from collections import deque

# Add project root to sys.path to allow importing app modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.core.face.detector import FaceDetector
from app.core.face.analyzer import FaceAnalyzer

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class RealTimeEmotionAnalyzer:
    """Class for real-time emotion analysis from video stream."""
    
    def __init__(self, camera_id: int = 0):
        """Initialize the emotion analyzer."""
        # Performance settings
        self.frame_skip = 7
        self.target_width = 400
        self.max_fps = 8
        self.cache_time = 0.8
        
        # Display settings
        self.font_scale = 0.35
        self.font_thickness = 1
        self.text_padding = 2
        self.line_spacing = 8
        self.min_confidence = 0.15  # Lowered from 0.3 to make it easier to display emotions
        
        # Text outline settings for better visibility
        self.outline_thickness = 2
        self.outline_color = (0, 0, 0)  # Black outline
        
        # Emotion colors (BGR format for OpenCV)
        self.emotion_colors = {
            'happy': (0, 255, 255),    # Yellow
            'sad': (255, 0, 0),        # Blue
            'angry': (0, 0, 255),      # Red
            'neutral': (255, 255, 255), # White
            'surprise': (255, 165, 0),  # Orange
            'fear': (128, 0, 128),     # Purple
            'disgust': (0, 255, 0)     # Green
        }
        
        # Initialize components
        self.detector = FaceDetector()
        self.analyzer = FaceAnalyzer()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Performance monitoring
        self.frame_times = []
        self.last_analysis_result = None
        self.last_analysis_time = 0
        
        # Emotion tracking - increased history length for more stability
        self.emotion_history = deque(maxlen=10)  # Increased from 5 to 10
        self.emotion_streak = 0
        self.last_emotion = None
        self.emotion_changes = 0
        
        # Session statistics
        self.session_start = time.time()
        self.emotion_counts = {emotion: 0 for emotion in self.emotion_colors.keys()}
        self.last_emotion_time = time.time()
        self.emotion_durations = {emotion: 0 for emotion in self.emotion_colors.keys()}
        
        logger.info("Real-time emotion analyzer initialized")
    
    def _process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[Dict]]:
        """Process a single frame."""
        try:
            # Resize frame
            height, width = frame.shape[:2]
            scale = self.target_width / width
            new_size = (self.target_width, int(height * scale))
            frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)
            
            # Detect faces
            faces = self.detector.detect_faces(frame)
            
            if not faces:
                logger.debug("No faces detected.")
                self._draw_status(frame, "No face detected", (0, 0, 255))
                self.last_analysis_result = None # Clear cache if no face
                self.last_analysis_time = 0
                # Update emotion duration for the last detected emotion before losing face
                if self.last_emotion is not None:
                    duration = time.time() - self.last_emotion_time
                    self.emotion_durations[self.last_emotion] += duration
                    self.last_emotion_time = time.time() # Reset time for next emotion detection

                # Reset emotion tracking if no face is detected
                self.emotion_history.clear()
                self.emotion_streak = 0
                self.last_emotion = None
                # Don't reset emotion_changes or emotion_counts for session stats
                
                return frame, None
            
            logger.debug(f"Detected {len(faces)} face(s).")

            # Process first face (assuming only one main subject)
            face = faces[0]
            x, y, w, h = face['bbox']
            
            # Check cache
            current_time = time.time()
            if (self.last_analysis_result and 
                current_time - self.last_analysis_time < self.cache_time):
                analysis_result = self.last_analysis_result
                logger.debug("Using cached analysis result.")
            else:
                face_img = frame[y:y+h, x:x+w]
                logger.debug("Performing new analysis.")
                analysis_result = self.analyzer.analyze_face(face_img)
                self.last_analysis_time = current_time
                self.last_analysis_result = analysis_result
                if analysis_result:
                    logger.debug(f"New analysis result: {analysis_result}")
                else:
                     logger.debug("New analysis returned None.")
            
            # Update emotion tracking
            if analysis_result and 'emotions' in analysis_result:
                emotions = analysis_result['emotions']
                if emotions:
                    top_emotion_item = max(emotions.items(), key=lambda x: x[1])
                    top_emotion = top_emotion_item[0]
                    top_confidence = top_emotion_item[1]
                    logger.debug(f"Top emotion: {top_emotion} with confidence {top_confidence:.2f}")

                    # Update emotion history and session stats based on detected emotion
                    current_emotion = top_emotion # Always use the current top emotion for tracking
                    self.emotion_history.append(current_emotion)

                    # Update emotion duration
                    if self.last_emotion is not None and current_emotion != self.last_emotion:
                         duration = time.time() - self.last_emotion_time
                         self.emotion_durations[self.last_emotion] += duration
                         self.last_emotion_time = time.time() # Reset time for new emotion
                    elif self.last_emotion is None:
                          self.last_emotion_time = time.time() # Start timer on first detection
                    
                    # Update emotion streak
                    if current_emotion == self.last_emotion:
                         self.emotion_streak += 1
                    else:
                         self.emotion_streak = 1
                         if self.last_emotion is not None: # Avoid counting change on first detection
                             self.emotion_changes += 1
                    
                    self.last_emotion = current_emotion # Update last_emotion after duration calculation
                    self.emotion_counts[current_emotion] += 1 # Count all detected top emotions

                else:
                    logger.debug("Analysis result has no emotions.")
                    # Update emotion duration for the last detected emotion before losing emotion data
                    if self.last_emotion is not None:
                         duration = time.time() - self.last_emotion_time
                         self.emotion_durations[self.last_emotion] += duration
                         self.last_emotion_time = time.time() # Reset time
                    
                    # Reset emotion tracking if no emotions are detected
                    self.emotion_history.clear()
                    self.emotion_streak = 0
                    self.last_emotion = None
            else:
                 logger.debug("Analysis result is None or has no 'emotions' key.")
                 # Update emotion duration for the last detected emotion before analysis failure
                 if self.last_emotion is not None:
                     duration = time.time() - self.last_emotion_time
                     self.emotion_durations[self.last_emotion] += duration
                     self.last_emotion_time = time.time() # Reset time

                 # Reset emotion tracking if analysis fails or no emotions key
                 self.emotion_history.clear()
                 self.emotion_streak = 0
                 self.last_emotion = None
            
            # Draw results
            self._draw_face_info(frame, face, analysis_result)
            
            return frame, analysis_result
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return frame, None
    
    def _draw_text_with_outline(self, frame: np.ndarray, text: str, position: Tuple[int, int], 
                              color: Tuple[int, int, int], scale: float = None) -> None:
        """Draw text with outline for better visibility."""
        if scale is None:
            scale = self.font_scale
            
        x, y = position
        
        # Draw outline in all 8 directions
        outline_color_bgr = self.outline_color # Assuming outline_color is already BGR
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue # Skip the center
                cv2.putText(frame, text, (x + dx * self.outline_thickness, y + dy * self.outline_thickness), font,
                            scale, outline_color_bgr, self.outline_thickness)
        
        # Draw the main text on top
        cv2.putText(frame, text, position, font,
                  scale, color, self.font_thickness)
    
    def _draw_face_info(self, frame: np.ndarray, face: Dict, analysis: Optional[Dict]) -> None:
        """Draw face information on the frame."""
        try:
            x, y, w, h = face['bbox']
            quality = face.get('quality', 0.5)
            
            logger.debug(f"Drawing face info for bbox {x,y,w,h}")

            # --- Determine emotions and confidence to display ---
            display_emotion = "Analyzing..."
            display_confidence = 0.0
            current_top_emotion = None
            current_top_confidence = 0.0
            
            # Get current frame's top emotion and confidence
            if analysis and 'emotions' in analysis and analysis['emotions']:
                current_emotions = analysis['emotions']
                # Find the top emotion and its confidence in the current frame
                current_top_emotion_item = max(current_emotions.items(), key=lambda item: item[1])
                current_top_emotion = current_top_emotion_item[0]
                current_top_confidence = current_top_emotion_item[1]
                logger.debug(f"Current frame top emotion: {current_top_emotion} ({current_top_confidence:.2f})")

            # Get dominant emotion from history (for smoothing)
            history_dominant_emotion = None
            history_confidence = 0.0
            if len(self.emotion_history) >= 3:  # Require at least 3 frames for history
                emotion_counts = {}
                for emotion in self.emotion_history:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                
                history_dominant_emotion_item = max(emotion_counts.items(), key=lambda item: item[1])
                history_dominant_emotion = history_dominant_emotion_item[0]
                history_confidence = history_dominant_emotion_item[1] / len(self.emotion_history)
                logger.debug(f"Historical dominant emotion: {history_dominant_emotion} ({history_confidence:.2f})")

            # Determine which emotion and confidence to display
            # Prioritize current frame emotion if it's above a reasonable threshold
            if current_top_emotion and current_top_confidence >= self.min_confidence:  # Use base threshold for current
                display_emotion = current_top_emotion
                display_confidence = current_top_confidence
                logger.debug(f"Displaying current top emotion (high confidence): {display_emotion} ({display_confidence:.2f})")
            # Otherwise, use the historical dominant emotion if it's stable enough
            elif history_dominant_emotion and history_confidence >= self.min_confidence * 0.8:  # Slightly lower threshold for history
                display_emotion = history_dominant_emotion
                display_confidence = history_confidence
                logger.debug(f"Displaying historical dominant emotion: {display_emotion} ({display_confidence:.2f})")
            # Fallback to current frame emotion if it's detected at all
            elif current_top_emotion and current_top_confidence > 0.0:
                display_emotion = current_top_emotion
                display_confidence = current_top_confidence
                logger.debug(f"Displaying current top emotion (low confidence fallback): {display_emotion} ({display_confidence:.2f})")

            # --- Draw face box and info ---
            
            # Get color based on displayed emotion or default
            color = self.emotion_colors.get(display_emotion, (255, 255, 255))  # White default
            
            # Draw face box with determined color
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Calculate base vertical position above the bounding box
            base_text_y = y - 5
            text_y_offset = 0

            # Draw quality indicator
            quality_text = f"Quality: {quality:.2f}"
            quality_color = (0, 255, 0) if quality >= 0.5 else (0, 165, 255)  # Green for Good, Orange for Poor
            self._draw_text_with_outline(frame, quality_text, (x, base_text_y + text_y_offset), quality_color, self.font_scale * 0.8)
            text_y_offset -= self.line_spacing + self.text_padding

            # Draw emotion text
            if display_emotion and display_emotion != "Analyzing...":
                logger.debug(f"Drawing emotion text: {display_emotion}")
                
                # Use streak only if displaying historical dominant emotion
                is_historical_display = (display_emotion == history_dominant_emotion and history_confidence >= self.min_confidence * 0.8)
                streak_text = f" ({self.emotion_streak}s)" if is_historical_display and self.emotion_streak > 1 else ""
                emotion_text = f"{display_emotion.title()}{streak_text}"
                self._draw_text_with_outline(frame, emotion_text, (x, base_text_y + text_y_offset), color, self.font_scale * 0.9)
                text_y_offset -= self.line_spacing + self.text_padding
                
                # Draw confidence
                conf_text = f"Confidence: {display_confidence:.0%}"
                self._draw_text_with_outline(frame, conf_text, (x, base_text_y + text_y_offset), (255, 255, 255), self.font_scale * 0.8)
                text_y_offset -= self.line_spacing + self.text_padding
            else:
                # Draw analyzing message
                logger.debug("Drawing 'Analyzing...'")
                self._draw_text_with_outline(frame, "Analyzing...", (x, base_text_y + text_y_offset), (255, 255, 255), self.font_scale * 0.9)
                text_y_offset -= self.line_spacing + self.text_padding

        except Exception as e:
            logger.error(f"Error drawing face info: {str(e)}")
    
    def _draw_session_stats(self, frame: np.ndarray) -> None:
        """Draw session statistics."""
        try:
            # Calculate session duration
            duration = time.time() - self.session_start
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            
            # Calculate most common emotion (Overall session)
            total_emotions = sum(self.emotion_counts.values())
            if total_emotions > 0:
                most_common = max(self.emotion_counts.items(), key=lambda x: x[1])
                emotion_percent = (most_common[1] / total_emotions) * 100
            else:
                most_common = ("none", 0)
                emotion_percent = 0
            
            # Draw stats in top-left corner
            x = 10 # Moved to the left side
            y = 25 # Keep at the top
            
            # Draw each stat with outline and slightly larger, higher quality font
            stats = [
                f"Time: {minutes:02d}:{seconds:02d}",
                f"Changes: {self.emotion_changes}",
                f"Overall Most: {most_common[0].title()} ({emotion_percent:.0f}%)" # Added Overall
            ]
            
            # Use a slightly larger scale and thickness for stats
            stats_font_scale = self.font_scale * 1.0 # Increased font size
            stats_font_thickness = self.font_thickness + 1 # Increased thickness
            line_height = 20 # Adjusted vertical spacing

            for i, stat in enumerate(stats):
                # Manually draw outline and text with specified font scale and thickness
                text_pos = (x, y + i * line_height)
                
                # Draw outline
                outline_color_bgr = self.outline_color
                font = cv2.FONT_HERSHEY_SIMPLEX
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        cv2.putText(frame, stat, (text_pos[0] + dx * (stats_font_thickness//2 + 1), text_pos[1] + dy * (stats_font_thickness//2 + 1)),
                                    font, stats_font_scale, outline_color_bgr, stats_font_thickness)

                # Draw main text
                cv2.putText(frame, stat, text_pos, font,
                            stats_font_scale, (255, 255, 255), stats_font_thickness)
            
        except Exception as e:
            logger.error(f"Error drawing session stats: {str(e)}")
    
    def _draw_status(self, frame: np.ndarray, message: str, color: Tuple[int, int, int]) -> None:
        """Draw status message on frame."""
        try:
            height, width = frame.shape[:2]
            
            # Calculate position (centered)
            text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 
                                      self.font_scale, self.font_thickness)[0]
            x = (width - text_size[0]) // 2
            y = height // 2
            
            # Draw text with outline
            self._draw_text_with_outline(frame, message, (x, y), color, self.font_scale * 0.9) # Slightly larger for status
            
        except Exception as e:
            logger.error(f"Error drawing status: {str(e)}")
    
    def run(self):
        """Run the emotion analyzer."""
        try:
            while True:
                start_time = time.time()
                
                # Grab frame
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to grab frame")
                    break
                
                # Process frame
                processed_frame, _ = self._process_frame(frame)
                
                # Calculate and display FPS
                self.frame_times.append(time.time() - start_time)
                if len(self.frame_times) > 10:
                    self.frame_times.pop(0)
                
                # Avoid division by zero if frame_times is empty
                if self.frame_times:
                    fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
                else:
                    fps = 0.0
                
                # Draw FPS
                self._draw_text_with_outline(processed_frame, f"FPS: {fps:.1f}", (10, 30), (0, 255, 0), self.font_scale * 0.8)
                
                # Display frame
                cv2.imshow('Emotion Analysis', processed_frame)
                
                # Check for exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Limit FPS
                time.sleep(max(0, 1.0/self.max_fps - (time.time() - start_time)))
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
        finally:
            self.cleanup()
    
    def generate_session_summary(self) -> None:
        """Generate and print a summary of the emotional session."""
        try:
            session_end = time.time()
            total_duration = session_end - self.session_start

            logger.info("\n--- Emotional Session Summary ---")
            logger.info(f"Session Duration: {int(total_duration)} seconds")
            logger.info(f"Total Emotion Changes: {self.emotion_changes}")
            
            if total_duration > 0:
                logger.info("Time spent in each emotion:")
                # Ensure the duration of the last detected emotion is added before summary
                if self.last_emotion is not None:
                    self.emotion_durations[self.last_emotion] += (session_end - self.last_emotion_time)
                    self.last_emotion_time = session_end # Update to avoid double counting if summary called multiple times

                sorted_emotions = sorted(self.emotion_durations.items(), key=lambda item: item[1], reverse=True)
                for emotion, duration in sorted_emotions:
                    if duration > 0:
                        percentage = (duration / total_duration) * 100
                        logger.info(f"  - {emotion.title()}: {int(duration)}s ({percentage:.1f}%)")
            else:
                logger.info("No emotional data recorded in this session.")

            logger.info("-------------------------------")

        except Exception as e:
            logger.error(f"Error generating session summary: {str(e)}")

    def cleanup(self):
        """Clean up resources and generate session summary."""
        try:
            # Generate session summary before releasing resources
            self.generate_session_summary()

            self.cap.release()
            cv2.destroyAllWindows()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

def main():
    """Main entry point."""
    try:
        camera_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
        analyzer = RealTimeEmotionAnalyzer(camera_id)
        analyzer.run()
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 