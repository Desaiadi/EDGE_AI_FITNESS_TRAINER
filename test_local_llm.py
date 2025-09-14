"""
Test script for Local LLM Manager
Tests workout and nutrition plan generation with NPU acceleration
"""

import logging
from local_llm_manager import LocalLLMManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def test_local_llm():
    """Test local LLM functionality"""

    # Initialize LLM Manager
    logging.info("Initializing Local LLM Manager...")
    llm_manager = LocalLLMManager()

    # Sample user profile for testing
    test_profile = {
        "name": "Test User",
        "age": 28,
        "weight": 75,  # kg
        "height": 175,  # cm
        "gender": "male",
        "activity_level": "moderate",
        "fitness_goal": "muscle_gain",
        "fitness_level": "intermediate",
    }

    # Test workout plan generation
    logging.info("\nGenerating workout plan...")
    try:
        workout_plan = llm_manager.generate_workout_plan(test_profile)
        logging.info("\n=== Workout Plan ===")
        logging.info(f"Daily Calories: {workout_plan['daily_calories']:.0f}")
        logging.info(f"BMR: {workout_plan['bmr']:.0f}")

        logging.info("\nWeekly Schedule:")
        for day, workout in workout_plan["weekly_schedule"].items():
            logging.info(f"\n{day}:")
            logging.info(f"Exercises: {', '.join(workout['exercises'])}")
            logging.info(f"Sets: {workout['sets']}")
            logging.info(f"Rest: {workout['rest_between_sets']}s")
            logging.info(f"Structure: {' → '.join(workout['structure'])}")

        logging.info("\nProgression Plan:")
        for week, plan in workout_plan["progression"].items():
            logging.info(f"{week}: {plan}")

    except Exception as e:
        logging.error(f"Failed to generate workout plan: {e}")

    # Test nutrition plan generation
    logging.info("\nGenerating nutrition plan...")
    try:
        nutrition_plan = llm_manager.generate_nutrition_plan(test_profile)
        logging.info("\n=== Nutrition Plan ===")
        logging.info(f"Target Calories: {nutrition_plan['target_calories']:.0f}")
        logging.info(f"Hydration Goal: {nutrition_plan['hydration_goal']:.0f}ml")

        logging.info("\nMacros:")
        for macro, amount in nutrition_plan["macros"].items():
            logging.info(f"{macro.title()}: {amount}g")

        logging.info("\nMeal Plan:")
        for meal, details in nutrition_plan["meal_plan"].items():
            logging.info(f"\n{meal.title()} ({details['calories']} cal):")
            logging.info(f"Foods: {', '.join(details['foods'])}")

    except Exception as e:
        logging.error(f"Failed to generate nutrition plan: {e}")


def test_npu_acceleration():
    """Test NPU acceleration setup"""
    logging.info("\nTesting NPU acceleration...")

    llm_manager = LocalLLMManager()

    # Check if NPU session was created successfully
    if llm_manager.session:
        providers = llm_manager.session.get_providers()
        logging.info(f"Active providers: {providers}")
        if "QNNExecutionProvider" in providers:
            logging.info("✅ NPU acceleration is active")
        else:
            logging.info("⚠️ NPU acceleration not available, using fallback providers")
    else:
        logging.warning("⚠️ Session creation failed, check NPU availability")


def main():
    """Run all tests"""
    logging.info("Starting Local LLM tests...")

    # Test NPU acceleration
    test_npu_acceleration()

    # Test plan generation
    test_local_llm()

    logging.info("\nTests completed!")


if __name__ == "__main__":
    main()
