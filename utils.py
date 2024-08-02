import matplotlib.pyplot as plt
import numpy as np
import imageio
import cv2


def clip_reward(reward):
    if reward < -1:
        return -1
    elif reward > 1:
        return 1
    else:
        return reward


# https://github.com/XinJingHao/PPO-Continuous-Pytorch
def action_adapter(a, max_a):
    return 2 * (a - 0.5) * max_a


def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84))
    frame = frame.astype(np.float32) / 255.0
    return frame


def save_sample_state(state):
    frame = preprocess_frame(state)
    plt.imshow(frame, cmap="gray")
    plt.savefig("sample_state.jpg")
    plt.close()


def save_animation(frames, filename):
    with imageio.get_writer(filename, mode="I", loop=0) as writer:
        for frame in frames:
            writer.append_data(frame)


def plot_running_avg(scores, env, metrics):
    avg_q_values = [m["average_q_value"] for m in metrics]
    run_avg_scores = np.zeros_like(scores)
    run_avg_qvals = np.zeros_like(avg_q_values)

    for i in range(len(scores)):
        run_avg_scores[i] = np.mean(scores[max(0, i - 100) : i + 1])
        run_avg_qvals[i] = np.mean(avg_q_values[max(0, i - 100) : i + 1])

    plt.figure(figsize=(10, 5))
    plt.plot(run_avg_scores, label="Average Score")
    plt.plot(run_avg_qvals, label="Average Q Value")
    plt.title("Running Avg Scores vs Qvals per 100 Games")
    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"metrics/{env}_running_avg_q.png")
    plt.close()
