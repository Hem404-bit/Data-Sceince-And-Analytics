def calculate_grade(marks):
    if marks >= 90:
        grade = "A"
        message = "Excellent! Keep up the great work"
    elif marks >= 80:
        grade = "B"
        message = "Very Good! Keep it up"
    elif marks >= 70:
        grade = "C"
        message = "Good job! You can do even better"
    elif marks >= 60:
        grade = "D"
        message = "Nice effort! Keep working hard"
    else:
        grade = "F"
        message = "Don't give up! Keep practicing and improve"

    return grade, message


# Get student name
name = input("Enter student name: ")

# Validate marks
while True:
    try:
        marks = float(input("Enter marks (0-100): "))

        if 0 <= marks <= 100:
            break
        else:
            print("Invalid marks! Please enter marks between 0 and 100.")

    except ValueError:
        print("Invalid input! Please enter a number.")


# Calculate grade
grade, message = calculate_grade(marks)


# Display result
print("RESULT FOR", name.upper() + ":")
print(f"Marks: {marks}/100")
print(f"Grade: {grade}")
print(f"Message: {message}")








