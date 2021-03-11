import time
from datetime import datetime
import csv
import os


def read_csv_file(filename):
    rows = []
    if os.path.exists(filename):
        with open(filename) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                rows.append(row)
    else:
        open(filename, "w+")
    return rows


# Stats filenames
csv_timestamps = "data/statistics/stats - timestamps.csv"
total_stats_filename = "data/statistics/stats - total numbers.csv"
dayly_stats_filename = "data/statistics/stats - dayly numbers.csv"
weekly_stats_filename = "data/statistics/stats - weekly numbers.csv"

# Stats headers
total_stats_header = [
    "",
    "mouthmask found detections:",
    "mouthmask not found detections:",
    "mouthmask found percentage:",
    "mouthmask not found percentage:",
]
dayly_stats_header = [
    "date",
    "mouthmask found detections:",
    "mouthmask not found detections:",
    "mouthmask found percentage:",
    "mouthmask not found percentage:",
]
weekly_stats_header = [
    "weeknumber",
    "mouthmask found detections:",
    "mouthmask not found detections:",
    "mouthmask found percentage:",
    "mouthmask not found percentage:",
]

# Read stats old data
total_stats = read_csv_file(total_stats_filename)
dayly_stats = read_csv_file(dayly_stats_filename)
weekly_stats = read_csv_file(weekly_stats_filename)


def update_stats(face):
    global total_stats_filename, csv_timestamps
    # prepare stats
    now = datetime.now()
    current_time = now.isoformat(timespec="seconds")
    week_num = now.isocalendar()[1]
    date, time = current_time.split("T")
    date_time = "{0} {1}".format(date, time)
    if face.mask_detected == True:
        add_with_mask = 1
        add_without_mask = 0
    elif face.mask_detected == False:
        add_without_mask = 1
        add_with_mask = 0

    # Update stats
    update_timestamps(date_time, face)
    update_last_row_of_stats(
        total_stats,
        total_stats_filename,
        total_stats_header,
        "total",
        add_with_mask,
        add_without_mask,
    )
    update_last_row_of_stats(
        dayly_stats,
        dayly_stats_filename,
        dayly_stats_header,
        date,
        add_with_mask,
        add_without_mask,
    )
    update_last_row_of_stats(
        weekly_stats,
        weekly_stats_filename,
        weekly_stats_header,
        week_num,
        add_with_mask,
        add_without_mask,
    )


def append_to_csv_file(filename, row):
    with open(filename, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row)


def write_to_csv_file(filename, rows):
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for row in rows:
            writer.writerow(row)


def update_timestamps(date_time, face):
    append_to_csv_file(csv_timestamps, [date_time, face.mask_detected])


def update_last_row_of_stats(
    stats, filename, header, column1, add_with_mask, add_without_mask
):
    old_column1 = column1
    old_with_mask = 0
    old_without_mask = 0
    # Making sure format is correct:
    if len(stats) == 0:
        stats.append(header)
    elif stats[0] != header:
        stats[0] = header
    if len(stats) == 1:
        stats.append([old_column1, old_with_mask, old_without_mask, "0%", "0%"])
    while len(stats) > 2 and stats[-1] == []:
        stats.pop(-1)
    if len(stats) > 1 and stats[-1] != []:
        old_with_mask = int(stats[-1][1])
        old_without_mask = int(stats[-1][2])
        old_column1 = stats[-1][0]

    with_mask = old_with_mask + add_with_mask
    without_mask = old_without_mask + add_without_mask

    total = with_mask + without_mask
    with_mask_percentage = str(int(round(with_mask / total * 100, 0))) + "%"
    without_mask_percentage = str(int(round(without_mask / total * 100, 0))) + "%"

    stats[-1] = [
        column1,
        with_mask,
        without_mask,
        with_mask_percentage,
        without_mask_percentage,
    ]
    if column1 == old_column1:
        write_to_csv_file(filename, stats)
    else:
        append_to_csv_file(filename, stats[-1])
