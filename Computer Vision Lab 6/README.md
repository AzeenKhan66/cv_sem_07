# Landmark Recognition Using the SIFT Algorithm

A computer vision lab that recognizes famous landmarks from a query photo using SIFT (Scale-Invariant Feature Transform) keypoints, Brute-Force matching with Lowe's ratio test, RANSAC homography verification, and landmark localization. The whole system is one Google Colab notebook.

## Contents

```
.
├── Landmark_Recognition_SIFT_Lab.ipynb   # the complete lab (run in Google Colab)
└── README.md
```

When the notebook runs, it creates this folder layout under `/content/landmark_lab/`:

```
landmark_lab/
├── landmark_db/            # reference images, one folder per landmark
│   ├── Eiffel Tower/
│   ├── Taj Mahal/
│   ├── Colosseum/
│   ├── Statue of Liberty/
│   └── Great Wall of China/
├── test_images/            # held-out photos used only for evaluation
│   ├── Eiffel Tower/ ... Great Wall of China/
│   └── Not in database/    # a landmark absent from the database (Pyramid of Giza)
├── downloads/              # raw downloads before splitting
├── query_images/           # your own uploaded query photos
└── results/
    ├── feature_db.pkl      # SIFT descriptors and keypoints with landmark labels
    └── test_results.csv    # Task 6 evaluation table
```

## Requirements

- Google Colab (CPU runtime is enough). The OpenCV 4.x build that ships with Colab includes `cv2.SIFT_create`.
- Internet access for the automatic image download from Wikimedia Commons, or photos uploaded by hand.
- Libraries: `opencv`, `numpy`, `pandas`, `matplotlib`. All are preinstalled in Colab.

## How to run

1. In Colab, choose **File → Upload notebook** and select `Landmark_Recognition_SIFT_Lab.ipynb`.
2. Choose **Runtime → Run all**.
3. The download cell fetches 5 reference and 2 test images per landmark, plus 2 images of an unknown landmark. This takes about a minute.
4. Results appear inline: the descriptor summary, keypoint visualizations, the demo query, the Task 6 table, and the accuracy summary.
5. To classify your own photo, run the optional upload cell and select an image.

### Manual dataset (if downloading is blocked)

Put about 5 reference photos per landmark in `/content/landmark_lab/landmark_db/<Landmark Name>/`, and 2 held-out test photos per landmark in `/content/landmark_lab/test_images/<Landmark Name>/`. Use the folder names exactly as listed above, then skip the download cell.

## Pipeline

| Step | Task | What happens |
|---|---|---|
| 1 | Database | Images are grouped by landmark folder. The folder name is the label. |
| 2 | SIFT extraction | Each reference image is converted to grayscale and processed with SIFT (up to 2000 keypoints, 128-D descriptors). Keypoints are drawn for review, and descriptors are saved with their labels. |
| 3 | Query processing | The query goes through the same grayscale and SIFT steps, and its keypoints are displayed. |
| 4 | Matching and recognition | Brute-Force matcher with Euclidean (L2) distance, `knnMatch(k=2)`, and Lowe's ratio test (0.75). The survivors are the *reliable matches*. Each landmark is scored by its best reference image, and landmarks are ranked by RANSAC inliers. |
| 5 | Localization | A RANSAC homography maps the reference corners into the query. The polygon is drawn only if the geometry is valid (finite, not too small, convex, not flipped, centre inside the image). Otherwise a message is shown. |
| 6 | Evaluation | Five conditions per landmark (original, rotated 45°, resized 50%, brightness ×0.4, different viewpoint) plus unknown-landmark queries. Each row records predicted landmark, keypoints, reliable matches, inliers, decision, correctness, and processing time. |

### Decision rule

A query is **recognized** as the top-ranked landmark only if all three conditions hold:

1. The landmark has at least `MIN_GOOD_MATCHES` (12) reliable matches.
2. Its best reference image has at least `MIN_INLIERS` (10) RANSAC inliers.
3. It leads the second-best landmark by at least `MIN_INLIER_MARGIN` (5) inliers.

Otherwise the query is **rejected** as an unknown landmark. A landmark is never chosen on raw match count alone.

## Configuration

All parameters are in the configuration cell:

| Parameter | Value | Meaning |
|---|---|---|
| `IMAGES_PER_LANDMARK_DB` | 5 | Reference images per landmark |
| `IMAGES_PER_LANDMARK_TEST` | 2 | Held-out test images per landmark |
| `MAX_SIDE` | 800 | Longest side in pixels after resizing |
| `SIFT_FEATURES` | 2000 | Keypoint cap per image |
| `LOWE_RATIO` | 0.75 | Lowe's ratio test threshold |
| `MIN_GOOD_MATCHES` | 12 | Minimum reliable matches |
| `MIN_INLIERS` | 10 | Minimum RANSAC inliers |
| `MIN_INLIER_MARGIN` | 5 | Required lead over the second-best landmark |
| `RANSAC_REPROJ_THRESH` | 5.0 | RANSAC reprojection threshold (pixels) |

## Results

Run in Google Colab (OpenCV 5.0.0), with 5 reference images per landmark and 25 reference images in total. Each query was run through the full pipeline.

### Accuracy by condition

| Condition | Correct / total | Accuracy |
|---|---|---|
| Original | 3 / 5 | 60% |
| Rotated 45° | 3 / 5 | 60% |
| Resized 50% | 3 / 5 | 60% |
| Reduced brightness | 3 / 5 | 60% |
| Different viewpoint | 1 / 5 | 20% |
| Landmark not in database (rejected) | 2 / 2 | 100% |

Overall: **13 / 25** known-landmark queries correct (52%), **0** wrong-landmark predictions (no false accepts), **2 / 2** unknown-landmark queries rejected. Mean processing time is **2.4 s per query**.

### Per-landmark outcome

| Landmark | Correct (of 5) | Notes |
|---|---|---|
| Great Wall of China | 5 | Recognized in all conditions, including the different-viewpoint photo (13 inliers) |
| Taj Mahal | 4 | Recognized in the original, rotated, resized and dark conditions; rejected for the different viewpoint |
| Statue of Liberty | 4 | Recognized in the original, rotated, resized and dark conditions; rejected for the different viewpoint |
| Colosseum | 0 | Rejected in every condition. Predicted as Colosseum but with 6–7 inliers, below the threshold of 10 |
| Eiffel Tower | 0 | Rejected in every condition. Predicted as Taj Mahal, Statue of Liberty or Great Wall, with at most 6 inliers |

The demo query (a held-out Eiffel Tower photo) was rejected with 6 inliers, which is below the threshold of 10.

A user-uploaded photo (`WhatsApp Image 2026-10-15…`) was rejected with 8 inliers against a closest candidate of Statue of Liberty. Its true landmark is not recorded in the notebook.

### Interpretation

* **The system never gives a wrong landmark name.** Every failure was a rejection, which is the safer error for a tourism system. Zero false accepts came from 25 known and 2 unknown queries.
* **Rotation and resizing did not hurt.** Rotated and resized versions were recognized as often as the originals for the landmarks that worked, which fits SIFT's scale and rotation invariance.
* **Brightness reduction did not hurt the recognized landmarks** (Taj Mahal and Statue of Liberty both kept matching at ×0.4 brightness).
* **Different viewpoint was the hardest condition** (1 of 5). A planar homography is a weak model for a 3-D structure seen from a different angle.
* **The thresholds are the main cause of the Eiffel Tower and Colosseum failures.** Their reliable-match counts (8–31) and inliers (0–7) were often close to the cutoffs, so the RANSAC inlier threshold of 10 rejected them. In the Colosseum "Different viewpoint" case, Eiffel Tower ranked first with 7 inliers and Colosseum was not chosen, so the query was rejected.
* **Poor reference coverage is a likely cause.** Each landmark has only 5 reference images, drawn from web search results with no control over the view. The Eiffel Tower set was not checked for relevance or viewpoint, and the notebook's review cell was not inspected in this run. This is a hypothesis to verify, not a confirmed cause.
* **The 2000-keypoint cap may limit matching** for the landmarks that failed. For both Eiffel Tower and Colosseum, three of the five query conditions reached the cap, so some keypoints may have been dropped.

## Possible improvements

* Add more reference images per landmark, especially for Eiffel Tower and Colosseum, and check that each one shows the landmark clearly.
* Lower `MIN_INLIERS` (for example, to 6–8) and measure the effect on false accepts with the unknown-landmark set. This is the main trade-off in this lab.
* Raise `SIFT_FEATURES` or the image size (`MAX_SIDE`) to keep more keypoints in large, detailed photos.
* Use a multi-homography or sequence-based check for viewpoint changes, or record one reference per viewpoint.
* Report the threshold sweep (accuracy against false accepts) in the Discussion section.

## Limitations

* **Planar assumption:** the homography treats each landmark as a flat surface. Large viewpoint changes on 3-D structures make the polygon inaccurate, so the geometry check rejects those cases instead of drawing them.
* **Dataset dependence:** accuracy depends on how many, and how varied, the reference photos are. Downloaded images can be irrelevant, so the review cell should be checked.
* **Lighting and texture:** very dark or overexposed images lose keypoints. Plain sky or low-texture scenes give few features.
* **Speed:** matching runs once per reference image, so processing time grows linearly with database size. It is fine for a small demo but would need an index (for example, a FLANN matcher or a vocabulary tree) for large databases.
* **Small sample:** the evaluation uses 25 known-landmark and 2 unknown-landmark queries, so the percentages are noisy. The thresholds were chosen by hand for this database and were not tuned on a separate validation set.

## References

- D. G. Lowe, "Distinctive Image Features from Scale-Invariant Keypoints," *International Journal of Computer Vision*, 2004.
- OpenCV documentation: SIFT, feature matching, `cv2.findHomography`.
