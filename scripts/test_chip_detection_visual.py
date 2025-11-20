#!/usr/bin/env python3

import cv2
import numpy as np
import os

# Just set this to your image file
image_path = "captured_image.jpg"

def detect_chips_multi_scale(cv_image, block_sizes=[5, 7, 9, 11], kernel_sizes=[3, 5, 7]):
    """
    Robust multi-scale chip detection that tries multiple parameter combinations
    and combines the results to handle different lighting/position conditions.
    """
    all_detections = []
    
    for block_size in block_sizes:
        for kernel_size in kernel_sizes:
            # --- Enhanced Preprocessing ---
            # 1. Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # 2. Bilateral filter to reduce noise while preserving edges
            blurred = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # 3. Adaptive Thresholding
            C_val = 2
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, block_size, C_val)
            
            # 4. Morphological operations
            kernel = np.ones((3,3), np.uint8)
            thresh_closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh_opened = cv2.morphologyEx(thresh_closed, cv2.MORPH_OPEN, kernel)
            
            # Additional closing to handle reflection gaps
            kernel_close = np.ones((kernel_size,kernel_size), np.uint8)
            thresh_final = cv2.morphologyEx(thresh_opened, cv2.MORPH_CLOSE, kernel_close)
            
            # 5. Find contours
            contours, hierarchy = cv2.findContours(thresh_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter out large contours
            filtered_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 4000:
                    filtered_contours.append(contour)
            
            contours = filtered_contours
            
            # --- Contour filtering ---
            min_chip_area = 300
            max_chip_area = 2000
            epsilon_factor = 0.06
            expected_aspect_ratio = 2.0
            aspect_ratio_tolerance = 0.7
            
            # Process each contour
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Area filter
                if not (min_chip_area < area < max_chip_area):
                    continue
                
                # Shape approximation
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon_factor * peri, True)
                
                # Must have 4 vertices
                if len(approx) != 4:
                    continue
                
                # Get bounding rectangle
                rect = cv2.minAreaRect(contour)
                (center_x_px, center_y_px) = rect[0]
                (w_px, h_px) = rect[1]
                angle_deg = rect[2]
                
                # Handle orientation
                if w_px < h_px:
                    angle_deg = angle_deg + 90.0
                    actual_width_px = h_px
                    actual_height_px = w_px
                else:
                    angle_deg = angle_deg
                    actual_width_px = w_px
                    actual_height_px = h_px
                
                # Normalize angle
                if angle_deg > 90.0:
                    angle_deg -= 180.0
                if angle_deg < -90.0:
                    angle_deg += 180.0
                
                if actual_width_px <= 0 or actual_height_px <= 0:
                    continue
                
                # Aspect ratio check
                current_aspect_ratio = actual_width_px / actual_height_px
                if not (expected_aspect_ratio - aspect_ratio_tolerance < current_aspect_ratio < expected_aspect_ratio + aspect_ratio_tolerance):
                    continue
                
                # Convexity check
                if not cv2.isContourConvex(approx):
                    continue
                
                # All checks passed - this is a chip!
                detection = {
                    'center': (center_x_px, center_y_px),
                    'area': area,
                    'angle': angle_deg,
                    'aspect_ratio': current_aspect_ratio,
                    'block_size': block_size,
                    'kernel_size': kernel_size,
                    'contour': contour
                }
                all_detections.append(detection)
    
    return all_detections

def merge_detections(detections, distance_threshold=30):
    """
    Merge duplicate detections from different parameter combinations.
    Uses clustering to group nearby detections and select the best one from each group.
    """
    if not detections:
        return []
    
    # Group detections by proximity
    merged = []
    used = set()
    
    for i, det1 in enumerate(detections):
        if i in used:
            continue
            
        # Find all detections close to this one
        group = [det1]
        used.add(i)
        
        for j, det2 in enumerate(detections):
            if j in used:
                continue
                
            # Calculate distance between centers
            center1 = det1['center']
            center2 = det2['center']
            distance = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
            
            if distance < distance_threshold:
                group.append(det2)
                used.add(j)
        
        # Select the best detection from this group
        # Prefer detections with better aspect ratio (closer to 2.0) and higher confidence
        best_detection = min(group, key=lambda x: abs(x['aspect_ratio'] - 2.0))
        
        # Add confidence score based on how many parameter combinations found this detection
        best_detection['confidence'] = len(group)
        merged.append(best_detection)
    
    return merged

def filter_false_positives(detections, image_shape):
    """
    Filter out likely false positives based on various criteria.
    """
    filtered = []
    height, width = image_shape[:2]
    
    for det in detections:
        center_x, center_y = det['center']
        
        # Filter 1: Aspect ratio - be more strict
        if not (1.3 < det['aspect_ratio'] < 3.0):  # Tighter bounds
            continue
            
        # Filter 2: Area bounds - be more strict
        if not (350 < det['area'] < 1800):  # Tighter area bounds
            continue
            
        # Filter 3: Position filtering - exclude near edges
        edge_margin = 50  # pixels from edge
        if (center_x < edge_margin or center_x > width - edge_margin or 
            center_y < edge_margin or center_y > height - edge_margin):
            continue
            
        # Filter 4: Confidence filtering - prefer detections found by multiple methods
        if det['confidence'] < 2:  # Must be found by at least 2 parameter combinations
            continue
            
        filtered.append(det)
    
    return filtered

# Load image
if not os.path.exists(image_path):
    print(f"Error: Image file {image_path} not found!")
    exit()

cv_image = cv2.imread(image_path)
if cv_image is None:
    print(f"Error: Could not load image {image_path}")
    exit()

print("Running multi-scale chip detection...")

# Run multi-scale detection
all_detections = detect_chips_multi_scale(cv_image)
print(f"Found {len(all_detections)} total detections across all parameter combinations")

# Merge duplicate detections
merged_detections = merge_detections(all_detections)
print(f"After merging duplicates: {len(merged_detections)} unique chips")

# Filter false positives
filtered_detections = filter_false_positives(merged_detections, cv_image.shape)
print(f"After filtering false positives: {len(filtered_detections)} chips")

# Create visualization
output_image = cv_image.copy()

# Draw all filtered detections
for i, det in enumerate(filtered_detections):
    # Get contour points for bounding box
    rect = cv2.minAreaRect(det['contour'])
    box_points_float = cv2.boxPoints(rect)
    box_points_int = np.intp(box_points_float)
    
    # Draw green bounding box
    cv2.drawContours(output_image, [box_points_int], 0, (0, 255, 0), 2)
    
    # Draw red center point
    center_x, center_y = det['center']
    cv2.circle(output_image, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
    
    # Draw chip number
    cv2.putText(output_image, str(i+1), (int(center_x-10), int(center_y-10)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# Add text overlay with detection count
cv2.putText(output_image, f"Chips detected: {len(filtered_detections)}", (10, 30),
           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Show results
print(f"\nFinal chip detections:")
for i, det in enumerate(filtered_detections):
    print(f"  Chip {i+1}: Center=({int(det['center'][0])}, {int(det['center'][1])}), "
          f"Angle={det['angle']:.2f}, Area={det['area']:.2f}, "
          f"AR={det['aspect_ratio']:.2f}, Confidence={det['confidence']}")

# Display the result
cv2.imshow("Multi-Scale Chip Detection", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"\nDetection complete! Found {len(filtered_detections)} chips.") 