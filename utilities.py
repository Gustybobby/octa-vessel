# utilities.py

import numpy as np
from sklearn.linear_model import LinearRegression
import math
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def fit_tangent(
    points: list[tuple[int, int]], eps: float = 1e-6
) -> tuple[float, float]:
    """
    Fits a tangent (line) to a given set of points using linear regression.
    """
    logger.debug("Fitting tangent to points")
    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]

    if np.std(x) < eps:
        x_const = np.mean(x)
        logger.debug("Detected vertical tangent")
        return (None, x_const)

    model = LinearRegression().fit(x.reshape(-1, 1), y)
    m = model.coef_[0]
    b = model.intercept_
    logger.debug(f"Tangent fitted with slope {m} and intercept {b}")
    return (m, b)


def calculate_angle(slope1: float, slope2: float) -> float:
    """
    Calculates the (acute) angle between two lines given their slopes.
    """
    logger.debug("Calculating angle between two slopes")

    def slope_to_angle(slope):
        if slope is None:
            return math.pi / 2.0
        return math.atan(slope)

    angle1 = slope_to_angle(slope1)
    angle2 = slope_to_angle(slope2)

    angle_rad = abs(angle2 - angle1)
    angle_deg = math.degrees(angle_rad)

    if angle_deg > 180:
        angle_deg = 360 - angle_deg
    if angle_deg > 90:
        angle_deg = 180 - angle_deg

    logger.debug(f"Calculated angle: {angle_deg} degrees")
    return angle_deg


def fit_tangents_at_junction(
    points_dict: dict,
    path1: str,
    path2: str,
    vessel_segments: dict,
    num_points: int = 3,
    display_plot: bool = False,
) -> float:
    """
    Fits tangents to two vessel segments and calculates the angle between them at a junction.

    Args:
        points_dict (dict): Dictionary containing the start/end points of each path.
                            For example: points_dict["3 - 8"] might store path data.
        path1 (str): The first vessel segment (e.g., "3 - 8").
        path2 (str): The second vessel segment (e.g., "7 - 2").
        num_points (int): The number of points to consider from each end of the path.

    Returns:
        float: The angle between the two tangents in degrees.
    """
    # Retrieve data from your dictionary
    # NOTE: Adjust this part as needed to match your dictionary structure and vessel_segments
    segment1 = ""
    segment2 = ""
    points1 = ""
    points2 = ""

    # Reverse paths in case they are not found directly
    path1_reverse = " - ".join(path1.split(" - ")[::-1])
    path2_reverse = " - ".join(path2.split(" - ")[::-1])
    path_end1 = path1[path1.find("-") + 2 :]  # everything after the dash/space
    path_end2 = path2[: path2.find(" ")]  # everything before the dash/space

    if path1 not in points_dict:
        points1 = points_dict[path1_reverse][path_end1]
        segment1 = vessel_segments[path1_reverse]
    else:
        points1 = points_dict[path1][path_end1]
        segment1 = vessel_segments[path1]

    if path2 not in points_dict:
        points2 = points_dict[path2_reverse][path_end2]
        segment2 = vessel_segments[path2_reverse]
    else:
        points2 = points_dict[path2][path_end2]
        segment2 = vessel_segments[path2]

    # Combine the segments for visualization
    if display_plot:
        plt.figure()
        plt.title(f"Paths: {path1}, {path2}")
        combined_segment = np.maximum(segment1, segment2)
        plt.imshow(combined_segment, cmap="gray", alpha=0.5)

    # Fit tangent lines
    tangent1_slope, tangent1_intercept = fit_tangent(points1)
    tangent2_slope, tangent2_intercept = fit_tangent(points2)

    # Plot the tangent lines
    # 1) If slope is None => vertical line at x = intercept
    # 2) If slope is not None => line is y = slope * x + intercept
    if display_plot:
        if tangent1_slope is None:
            plt.axvline(x=tangent1_intercept, color="r", label=f"Tangent 1: {path1}")
        else:
            # Create a range of x for path1
            x_min_1 = (
                min(np.array(points1)[:, 0].min(), np.array(points2)[:, 0].min()) - 10
            )
            x_max_1 = (
                max(np.array(points1)[:, 0].max(), np.array(points2)[:, 0].max()) + 10
            )
            x_vals_1 = np.linspace(x_min_1, x_max_1, 100)
            y1_vals = tangent1_slope * x_vals_1 + tangent1_intercept
            plt.plot(x_vals_1, y1_vals, "r-", label=f"Tangent 1: {path1}")

        if tangent2_slope is None:
            plt.axvline(x=tangent2_intercept, color="b", label=f"Tangent 2: {path2}")
        else:
            # Create a range of x for path2
            x_min_2 = (
                min(np.array(points1)[:, 0].min(), np.array(points2)[:, 0].min()) - 10
            )
            x_max_2 = (
                max(np.array(points1)[:, 0].max(), np.array(points2)[:, 0].max()) + 10
            )
            x_vals_2 = np.linspace(x_min_2, x_max_2, 100)
            y2_vals = tangent2_slope * x_vals_2 + tangent2_intercept
            plt.plot(x_vals_2, y2_vals, "b-", label=f"Tangent 2: {path2}")

        # Scatter the actual points
        points1_np = np.array(points1)
        points2_np = np.array(points2)
        plt.scatter(
            points1_np[:, 0],
            points1_np[:, 1],
            color="red",
            label="Points from Path 1",
            zorder=5,
        )
        plt.scatter(
            points2_np[:, 0],
            points2_np[:, 1],
            color="blue",
            label="Points from Path 2",
            zorder=5,
        )

        plt.legend()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        plt.show()

    # Calculate the angle between the two tangents
    angle = calculate_angle(tangent1_slope, tangent2_slope)
    return angle
