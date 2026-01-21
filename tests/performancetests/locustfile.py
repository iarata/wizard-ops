import random
import io
from locust import HttpUser, LoadTestShape, between, task
from PIL import Image
import numpy as np

class FoodAnalyzerUser(HttpUser):
    """A locust user who checks health endpoint and posts an image. 1/10 times it posts an invalid image."""
    wait_time = between(1, 2)

    def on_start(self):
        """Creates test image (only once per user)."""
        img = Image.fromarray(np.random.randint(0,255, (224,224,3), dtype=np.uint8))
        self.img_bytes = io.BytesIO()
        img.save(self.img_bytes, format='JPEG')
        self.img_bytes.seek(0)

    @task(1)
    def get_root(self) -> None:
        """Test root endpoint."""
        self.client.get("/")

    @task(10)
    def analyze_normal_image(self):
        """Test normal image analysis."""
        self.img_bytes.seek(0)
        files = {'file': ('food.jpg', self.img_bytes, 'image/jpeg')}
        
        self.client.post("/analyze", files=files)

    # @task(1)
    # def analyze_invalid_file(self):
    #     """Test error handling."""



class LoadTest(LoadTestShape):
    """
    Load test with 

    Keyword arguments:

        stages -- A list of dicts, each representing a stage with the following keys:
            duration -- When this many seconds pass the test is advanced to the next stage
            users -- Total user count
            spawn_rate -- Number of users to start/stop per second
            stop -- A boolean that can stop that test at a specific stage

        stop_at_end -- Can be set to stop once all stages have run.
    """

    stages = [
        # Extended warmup: 0-120s (2 minutes)
        {"duration": 30, "users": 2, "spawn_rate": 1},
        {"duration": 60, "users": 5, "spawn_rate": 2},
        {"duration": 90, "users": 10, "spawn_rate": 3},
        {"duration": 120, "users": 10, "spawn_rate": 2}, 
        
        # Main test: 120-360s (4 minutes at steady loads)
        {"duration": 150, "users": 15, "spawn_rate": 5},   # 120-150s
        {"duration": 180, "users": 20, "spawn_rate": 5},   # 150-180s
        {"duration": 210, "users": 35, "spawn_rate": 5},   # 180-210s
        {"duration": 240, "users": 50, "spawn_rate": 5},   # 210-240s
        {"duration": 270, "users": 65, "spawn_rate": 5},   # 240-270s
        {"duration": 300, "users": 80, "spawn_rate": 5},   # 270-300s
        {"duration": 330, "users": 90, "spawn_rate": 5},   # 300-330s
        {"duration": 360, "users": 100, "spawn_rate": 5},  # 330-360s
        
        # Stress: 360-480s (2 minutes spike)
        {"duration": 370, "users": 150, "spawn_rate": 20}, # Quick spike
        {"duration": 480, "users": 150, "spawn_rate": 5},  # Hold at peak
        
        #Optional ramp down: 480-510s
        {"duration": 500, "users": 50, "spawn_rate": 10},
        {"duration": 510, "users": 10, "spawn_rate": 5},
     ]

    def tick(self):
        run_time = self.get_run_time()

        for stage in self.stages:
            if run_time < stage["duration"]:
                tick_data = (stage["users"], stage["spawn_rate"])
                return tick_data

        return None