import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import CLIPModel, CLIPProcessor
from pathlib import Path
from PIL import Image
import gradio as gr
import tqdm
import cv2
import time
import faiss
import json
import os