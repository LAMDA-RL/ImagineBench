import numpy as np
from tqdm import tqdm
from gymnasium import Env, Wrapper
from gymnasium.core import Any, WrapperObsType, WrapperActType
from metaworld.envs import reward_utils, ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE


num_tau = 1000
baseline_env_name_list = [
    # Basic skills
    'reach-v2-goal-observable',  # 1
    'push-v2-goal-observable',  # 1
    'pick-place-v2-goal-observable',  # 1
    # Easy-Level skills
    'button-press-v2-goal-observable',
    'door-unlock-v2-goal-observable',
    'door-open-v2-goal-observable',
    'window-open-v2-goal-observable',
    'faucet-open-v2-goal-observable',
    # Hard-Level pre-skills
    'coffee-push-v2-goal-observable',  # 2
    'coffee-button-v2-goal-observable',  # 2
    # duplicate with easy-level: 'door-unlock-v2-goal-observable',
    # duplicate with easy-level: 'door-open-v2-goal-observable',
]
rephrase_level_env_name_list = [
    'rep-reach-v2-goal-observable',
    'rep-push-v2-goal-observable',
    'rep-pick-place-v2-goal-observable',
    'rep-button-press-v2-goal-observable',
    'rep-door-unlock-v2-goal-observable',
    'rep-door-open-v2-goal-observable',
    'rep-window-open-v2-goal-observable',
    'rep-faucet-open-v2-goal-observable',
    'rep-coffee-push-v2-goal-observable',
    'rep-coffee-button-v2-goal-observable',
]
easy_level_env_name_list = [
    # Ability to adapt to the noisy wall
    'reach-wall-v2-goal-observable',  # 1
    'push-wall-v2-goal-observable',  # 1
    'pick-place-wall-v2-goal-observable',  # 1
    'button-press-wall-v2-goal-observable',
    # Contrast skills
    'door-lock-v2-goal-observable',
    'door-close-v2-goal-observable',
    'window-close-v2-goal-observable',
    'faucet-close-v2-goal-observable',
]
hard_level_env_name_list = [
    # Combination of pre-skills
    'make-coffee-v2-goal-observable',  # 2
    'locked-door-open-v2-goal-observable',
    # Generalizability
    'hammer-v2-goal-observable',
    'soccer-v2-goal-observable',  # 2
]


# Verify definition
env_name_list_to_verify = baseline_env_name_list + rephrase_level_env_name_list + easy_level_env_name_list + hard_level_env_name_list
for env_name in env_name_list_to_verify:
    assert env_name in ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE.keys()


COMMON_LENGTH = 7
special_entity_length_info = {
    'gripper': 4,
    'target_location': 3,
}
entity_list = [
    # Required entity
    'gripper',
    'target_location',
    # Optional entity
    'target_object',
    # Easy-Level
    'button',
    'lock',
    'handle',
    'window',
    'faucet',
    'wall',
    # Hard-level
    'coffee',
    'coffee-button',
    'hammer',
    'nail',
    'soccer',
]
entity2index = {}
current_index = 0
for entity in entity_list:
    if entity in special_entity_length_info.keys():
        length = special_entity_length_info[entity]
    else:
        length = COMMON_LENGTH
    entity2index[entity] = np.arange(current_index, current_index + length)

    current_index += length
offline_observation_dim = current_index
offline_action_dim = 4

num_nl = 20
llm_encoding_dim = 768
en2nl = {
    # ==== Baseline ====
    # Basic skills
    'reach-v2-goal-observable': [
        "Relocate the gripper to the designated spot.",
        "Position the gripper at the intended location.",
        "Transfer the gripper to the specified area.",
        "Shift the gripper towards the target point.",
        "Navigate the gripper to the predetermined location.",
        "Direct the gripper to the chosen destination.",
        "Guide the gripper to the target site.",
        "Transport the gripper to the aimed location.",
        "Propel the gripper towards the goal position.",
        "Advance the gripper to the target vicinity.",
        "Place the gripper at the marked location.",
        "Set the gripper in the target area.",
        "Slide the gripper to the intended spot.",
        "Reposition the gripper towards the designated location.",
        "Maneuver the gripper into the specified position.",
        "Command the gripper to the target destination.",
        "Adjust the gripper to the desired location.",
        "Align the gripper with the target point.",
        "Steer the gripper to the aimed spot.",
        "Dispatch the gripper to the predetermined position.",
    ],
    'push-v2-goal-observable': [
        "Employ the gripper to propel the target object towards its designated location.",
        "Utilize the gripper to advance the target object to its intended position.",
        "Employ the gripper mechanism to drive the target object to its specified location.",
        "Make use of the gripper to maneuver the target object to its designated spot.",
        "Operate the gripper to shove the target object to the desired location.",
        "Employ the gripper to thrust the target object towards its prescribed destination.",
        "Utilize the gripper to move the target object to the designated location.",
        "Use the gripper to nudge the target object to the required position.",
        "Employ the gripper to shift the target object to the predetermined location.",
        "Utilize the gripper to propel the target object to the assigned location.",
        "Operate the gripper to push the target object towards the intended spot.",
        "Employ the gripper to drive the target object to its predefined location.",
        "Utilize the gripper to advance the target object to its predetermined spot.",
        "Make use of the gripper to maneuver the target object to its specified destination.",
        "Operate the gripper to shove the target object to the designated spot.",
        "Employ the gripper to thrust the target object towards its intended location.",
        "Utilize the gripper to move the target object to the prescribed position.",
        "Use the gripper to nudge the target object to the designated location.",
        "Employ the gripper to shift the target object to the desired spot.",
        "Utilize the gripper to propel the target object to the predetermined destination.",
    ],
    'pick-place-v2-goal-observable': [
        "Employ the gripper to seize the designated item and transfer it to the specified position.",
        "Utilize the gripper for grasping the desired object and relocating it to the designated spot.",
        "Employ the gripper mechanism to grasp the object of interest and deposit it at the predetermined destination.",
        "Apply the gripper tool to grasp the target object and move it to the intended location.",
        "Use the gripper device to seize the specified item and position it at the desired place.",
        "Utilize the gripper apparatus to capture the designated object and place it at the specified location.",
        "Employ the gripper mechanism to clasp the target item and shift it to the designated site.",
        "Utilize the gripper for capturing the desired object and transferring it to the allocated location.",
        "Apply the gripper tool to grasp the specified object and relocate it to the intended position.",
        "Use the gripper device to seize the target object and move it to the designated spot.",
        "Employ the gripper apparatus to grasp the designated item and deposit it at the predetermined destination.",
        "Utilize the gripper mechanism to capture the desired object and position it at the intended location.",
        "Apply the gripper tool to seize the designated object and shift it to the specified site.",
        "Use the gripper device to grasp the target item and transfer it to the allocated location.",
        "Employ the gripper apparatus to capture the specified object and relocate it to the desired position.",
        "Utilize the gripper mechanism to grasp the target object and place it at the designated spot.",
        "Apply the gripper tool to capture the desired item and move it to the predetermined destination.",
        "Use the gripper device to grasp the designated object and position it at the intended location.",
        "Employ the gripper apparatus to seize the target item and shift it to the specified site.",
        "Utilize the gripper mechanism to capture the desired object and transfer it to the allocated position.",
    ],
    # Easy-Level contrast skill
    'button-press-v2-goal-observable': [
        "Utilize the gripper to firmly depress the button.",
        "Apply pressure with the gripper to activate the button.",
        "Engage the gripper to push down on the button.",
        "Employ the gripper to execute the action of pressing the button.",
        "Operate the gripper to exert force on the button.",
        "Manipulate the gripper to engage the button.",
        "Use the gripper to initiate contact with the button.",
        "Employ the gripper to make contact with and depress the button.",
        "Activate the button by means of the gripper.",
        "Utilize the gripper to trigger the button.",
        "Apply the gripper to engage the button in a downward motion.",
        "Employ the gripper to enact the pressing of the button.",
        "Execute the action of button pressing with the gripper.",
        "Utilize the gripper to apply force onto the button surface.",
        "Engage the gripper to initiate the button's activation.",
        "Employ the gripper to depress the button firmly.",
        "Use the gripper to push the button downward.",
        "Apply pressure on the button using the gripper's mechanism.",
        "Engage the gripper to interact with and push the button.",
        "Utilize the gripper's functionality to press down on the button.",
    ],
    'door-unlock-v2-goal-observable': [
        "Employ the gripper to turn the door's unlocking mechanism.",
        "Utilize the gripper to manipulate the lock and open the door.",
        "Engage the gripper to rotate the lock mechanism and release the door.",
        "Operate the gripper to unlock the door by twisting the key.",
        "Manipulate the gripper to disengage the door lock and grant access.",
        "Apply the gripper to the door's locking mechanism to unlock it.",
        "Utilize the gripper's dexterity to release the door latch.",
        "Activate the gripper to turn the key and unlock the door.",
        "Employ the gripper to interact with the door's locking mechanism and open it.",
        "Use the gripper to manipulate the lock mechanism and unlatch the door.",
        "Engage the gripper to operate the door's locking system and grant entry.",
        "Employ the gripper's functionality to unlock the door securely.",
        "Utilize the gripper to turn the keyhole and release the door lock.",
        "Apply pressure with the gripper to disengage the door's locking mechanism.",
        "Operate the gripper to unlock the door by turning the knob.",
        "Manipulate the gripper to unlock the door's deadlock securely.",
        "Utilize the gripper to twist and release the bolt, unlocking the door.",
        "Engage the gripper to rotate the lock cylinder and open the door.",
        "Use the gripper to manipulate the door handle and unlock it.",
        "Employ the gripper's precision to unlock the door with ease.",
    ],
    'door-open-v2-goal-observable': [
        "Utilize the gripper to grasp the door handle and pull it open.",
        "Employ the gripper to grip the door handle and swing it outward.",
        "Engage the gripper to seize the door knob and turn it, opening the door.",
        "Operate the gripper to clasp the door handle and pull it towards you.",
        "Manipulate the gripper to grasp the door's edge and pull it ajar.",
        "Apply the gripper to the door's handle, pulling it to unlock and open.",
        "Utilize the gripper's mechanism to latch onto the door's handle and pull.",
        "Activate the gripper to firmly grip the door's handle and push it open.",
        "Employ the gripper to seize the door knob and turn it, swinging the door open.",
        "Use the gripper to grasp the door's edge and pull it wide open.",
        "Engage the gripper to clench the door handle and pull it outward, opening the door.",
        "Apply pressure with the gripper to the door handle, pushing it open.",
        "Utilize the gripper to grasp the door knob and twist it, releasing the latch.",
        "Operate the gripper to firmly hold the door handle and pull it open smoothly.",
        "Manipulate the gripper to grip the door's edge and pull it towards you.",
        "Employ the gripper to firmly grasp the door knob and turn it, allowing the door to swing open.",
        "Activate the gripper to seize the door handle and pull it, releasing the latch.",
        "Use the gripper's dexterity to grip the door's edge and pull it open effortlessly.",
        "Engage the gripper to firmly clasp the door handle and pull it open with ease.",
        "Apply the gripper to the door's handle, exerting force to open it smoothly.",
    ],
    'window-open-v2-goal-observable': [
        "Employ the clamping tool to pry the window open.",
        "Utilize the grabbing device for window aeration.",
        "Activate the gripper to initiate the window's opening.",
        "Engage the gripper to unlatch and part the window.",
        "Manipulate the gripper to facilitate the window's aperture.",
        "Apply the gripper's mechanism to swing the window outward.",
        "Command the gripper to separate the window from its seal.",
        "Operate the gripper to achieve the window's opening.",
        "Handle the gripper to disengage the window's lock.",
        "Wield the gripper to create an opening in the window.",
        "Make use of the gripper to retract the window.",
        "Put the gripper into action to unfasten the window.",
        "Mobilize the gripper to slide the window open.",
        "Direct the gripper to hoist the window pane.",
        "Task the gripper with the job of opening the window.",
        "Assign the gripper to the duty of window manipulation.",
        "Instruct the gripper to execute the window's unveiling.",
        "Call upon the gripper to orchestrate the window's opening.",
        "Deploy the gripper to dislodge the window from its frame.",
        "Guide the gripper to perform the act of opening the window.",
    ],
    'faucet-open-v2-goal-observable': [
        "Employ the gripping tool to turn the faucet on.",
        "Utilize the clamp to twist the tap open.",
        "Activate the faucet by manipulating it with the gripper.",
        "With the gripper, commence the flow of water from the faucet.",
        "Apply the gripper to initiate the tap's operation.",
        "Engage the faucet's valve using the gripper.",
        "Operate the water spout by employing the gripper.",
        "Handle the gripper to release the faucet's stream.",
        "Make use of the gripper to unlock the flow from the faucet.",
        "Execute the opening of the faucet by using the gripper.",
        "Manipulate the gripper to achieve the faucet's activation.",
        "Employ the mechanical hand to twist open the water valve.",
        "Use the gripper as a tool to open up the water outlet.",
        "Take the gripper and turn the faucet to start the water flow.",
        "With the gripper in hand, proceed to open the water tap.",
        "Utilize the gripper's leverage to unseal the faucet.",
        "Command the faucet to open by applying the gripper.",
        "Grasp the faucet with the gripper and turn to release water.",
        "Engage the gripper to disengage the faucet's seal.",
        "With the gripper, perform the action of opening the water source.",
    ],
    # Hard-Level pre-skill
    'coffee-push-v2-goal-observable': [
        "Employ the gripper to nudge the coffee beneath the coffee machine.",
        "Utilize the gripper to slide the coffee under the coffee machine.",
        "Maneuver the gripper to push the coffee below the coffee machine.",
        "Operate the gripper to press the coffee underneath the coffee machine.",
        "Use the gripper to shove the coffee beneath the coffee machine.",
        "Apply the gripper to propel the coffee under the coffee machine.",
        "Deploy the gripper to move the coffee below the coffee machine.",
        "Employ the gripper to drive the coffee underneath the coffee machine.",
        "Utilize the gripper to advance the coffee under the coffee machine.",
        "Maneuver the gripper to thrust the coffee below the coffee machine.",
        "Operate the gripper to shift the coffee underneath the coffee machine.",
        "Use the gripper to guide the coffee under the coffee machine.",
        "Apply the gripper to position the coffee below the coffee machine.",
        "Deploy the gripper to relocate the coffee underneath the coffee machine.",
        "Employ the gripper to transfer the coffee under the coffee machine.",
        "Utilize the gripper to insert the coffee beneath the coffee machine.",
        "Maneuver the gripper to slip the coffee under the coffee machine.",
        "Operate the gripper to maneuver the coffee below the coffee machine.",
        "Use the gripper to place the coffee beneath the coffee machine.",
        "Apply the gripper to deposit the coffee under the coffee machine.",
    ],
    'coffee-button-v2-goal-observable': [
        "Utilize the gripper to depress the button on the coffee machine.",
        "Employ the gripper to push down the button of the coffee machine.",
        "Use the gripper to apply pressure to the button of the coffee machine.",
        "Operate the gripper to activate the button of the coffee machine.",
        "Employ the gripper to engage the button on the coffee machine.",
        "Utilize the gripper to exert force on the button of the coffee machine.",
        "Manipulate the gripper to trigger the button of the coffee machine.",
        "Utilize the gripper to operate the button of the coffee machine.",
        "Apply the gripper to the button of the coffee machine to initiate.",
        "Utilize the gripper to actuate the button of the coffee machine.",
        "Depress the button of the coffee machine using the gripper.",
        "Engage the gripper to press down the button of the coffee machine.",
        "Activate the button of the coffee machine by using the gripper.",
        "Push the button of the coffee machine using the gripper mechanism.",
        "Employ the gripper to push the button of the coffee machine down.",
        "Use the gripper to execute the pressing of the coffee machine's button.",
        "Apply pressure with the gripper to activate the coffee machine's button.",
        "Trigger the button on the coffee machine by manipulating the gripper.",
        "Press down on the button of the coffee machine with the gripper.",
        "Employ the gripper to operate the coffee machine's button by pressing it.",
    ],
    
    # ==== Rephrase-Level ====
    # Basic skills
    'rep-reach-v2-goal-observable': [
        "I'm dissatisfied with the gripper's current location; kindly adjust it to reach the desired position.",
        "The gripper's current placement doesn't suit me; could you relocate it to the target position?",
        "I'm not fond of where the gripper is placed now; could you shift it to the intended position?",
        "The gripper's present position isn't to my liking; please maneuver it to the target position.",
        "I'm displeased with the gripper's current stance; kindly adjust it to meet the target position.",
        "The current gripper location doesn't meet my preferences; could you move it to reach the desired position?",
        "I find the current gripper placement unsatisfactory; please relocate it to the target position.",
        "The gripper's current position is unfavorable to me; kindly move it to the desired location.",
        "I'm not content with where the gripper is currently situated; please adjust it to reach the target position.",
        "The gripper's current position doesn't suit my needs; could you please reposition it to reach the desired location?",
        "I genuinely wish for the successful relocation of the gripper to the specified destination.",
        "My earnest desire is for the gripper to reach the target location smoothly.",
        "I sincerely hope that the gripper can be moved to the intended location without any hindrance.",
        "It is my genuine aspiration that the gripper can be navigated to the desired target area.",
        "I earnestly desire that the gripper reaches the designated location without encountering any obstacles.",
        "My sincere hope is that the gripper can be transported to the target location promptly and efficiently.",
        "I genuinely wish for the successful movement of the gripper to the desired destination.",
        "It is my sincere desire that the gripper reaches the target location with ease.",
        "I genuinely hope for the gripper to be relocated to the specified target area in a timely manner.",
        "My earnest wish is for the gripper to be guided to the target location successfully.",
    ],
    'rep-push-v2-goal-observable': [
        "The current location of the target object isn't satisfactory to me; please utilize the gripper to nudge it to the target position.",
        "I'm not pleased with where the target object is currently situated; could you employ the gripper to guide it to the intended position?",
        "The target object's present placement doesn't meet my preferences; kindly use the gripper to push it to the desired position.",
        "I'm dissatisfied with the current position of the target object; could you please manipulate the gripper to push it to the target position?",
        "The current location of the target object isn't to my liking; please utilize the gripper to nudge it toward the target position.",
        "I find the current placement of the target object unsatisfactory; could you employ the gripper to move it to the desired position?",
        "The target object's current position doesn't meet my requirements; kindly use the gripper to push it to the target position.",
        "I'm not content with where the target object is currently positioned; please use the gripper to guide it to the desired location.",
        "The current position of the target object isn't suitable for me; could you manipulate the gripper to push it to the target position?",
        "I'm displeased with the current location of the target object; please use the gripper to nudge it to the desired position.",
        "Utilize the gripper mechanism to relocate the designated item to its intended destination.",
        "Employ the gripper to maneuver the specified object to the desired location.",
        "Engage the gripper device to transport the target item to its designated spot.",
        "Activate the gripper tool to push the object to the predetermined location.",
        "Operate the gripper apparatus to facilitate the movement of the target object to its specified destination.",
        "Employ the gripper mechanism in order to facilitate the transfer of the object to its designated position.",
        "Utilize the gripper system to navigate the specified object to the desired destination.",
        "Activate the gripper tool to guide the target item towards its intended location.",
        "Employ the gripper device to push the designated object to the prescribed destination.",
        "Utilize the gripper mechanism to facilitate the transportation of the target object to its predetermined location.",
    ],
    'rep-pick-place-v2-goal-observable': [
        "I have a negative sentiment towards the current placement of the object of interest; therefore, I intend to utilize the gripper mechanism to lift it and relocate it to the desired destination.",
        "The current arrangement of the designated item is unsatisfactory to me, prompting me to employ the gripper for the purpose of relocating it to the specified destination.",
        "I am not pleased with the current positioning of the target object; hence, I plan to employ the gripper tool to grasp it and transport it to the intended location.",
        "The current location of the object I'm targeting does not meet my preferences, so I'll be utilizing the gripper to lift it and transfer it to its desired location.",
        "I harbor a disapproval towards the present placement of the targeted object, prompting me to utilize the gripper mechanism to lift and reposition it to the desired spot.",
        "The current spatial arrangement of the target item does not align with my preferences, necessitating the use of the gripper to pick it up and place it in the designated location.",
        "I hold a disdain towards the current orientation of the object under consideration, compelling me to use the gripper to relocate it to the specified destination.",
        "I have an aversion towards the current positioning of the target object, leading me to deploy the gripper to grasp and transport it to the desired location.",
        "The current placement of the object I'm targeting is unsatisfactory to me, so I'm opting to use the gripper to lift and move it to the desired destination.",
        "I express a lack of approval towards the current position of the object of interest, motivating me to employ the gripper tool to pick it up and place it at the designated location.",
        "It would bring me great joy if you could utilize the gripper to grasp the target object and relocate it to the desired location.",
        "I'd be delighted if you could employ the gripper to lift the target object and position it at the designated spot.",
        "If you could use the gripper to snatch the target object and transfer it to the specified location, it would make me very happy.",
        "I would greatly appreciate it if you could manipulate the gripper to seize the target object and place it precisely where it needs to be, bringing me much happiness.",
        "Utilizing the gripper to grasp the target object and transport it to the desired location would fill me with joy.",
        "I'd be overjoyed if you could employ the gripper to pick up the target object and carefully place it at the designated location.",
        "Using the gripper to lift the target object and relocate it to the desired position would bring me immense happiness.",
        "I would be thrilled if you could utilize the gripper to grasp the target object and deposit it at the desired location, bringing me great satisfaction.",
        "Employing the gripper to pick up the target object and transfer it to the specified location would make me very happy indeed.",
        "It would make me extremely happy if you could use the gripper to seize the target object and place it precisely at the target location.",
    ],
    # Easy-Level contrast skill
    'rep-button-press-v2-goal-observable': [
        "I have a displeasure towards the inactive state of the button; therefore, I intend to utilize the gripper to apply pressure and activate it in order to open it.",
        "The current state of the button being inactive is not to my liking, prompting me to use the gripper to press it and initiate its function of opening.",
        "I am not satisfied with the button being in the off position, thus prompting me to employ the gripper mechanism to exert pressure and activate it for the purpose of opening.",
        "I am dissatisfied with the current non-functional state of the button, compelling me to use the gripper to press it and facilitate its opening.",
        "I hold a disapproval towards the button being inactive, prompting me to utilize the gripper mechanism to apply pressure and activate it in order to open.",
        "The current inactive state of the button does not align with my preferences, necessitating the use of the gripper to press it and initiate its opening function.",
        "I express a dislike towards the button being off, compelling me to use the gripper to apply pressure and activate it for the purpose of opening.",
        "I harbor an aversion towards the inactive state of the button, leading me to deploy the gripper to press it and facilitate its opening.",
        "The current off position of the button is unsatisfactory to me, so I'm opting to use the gripper to exert pressure and activate it in order to open.",
        "I am displeased with the button being off, motivating me to employ the gripper tool to press it and thereby open it.",
        "Could you kindly utilize the gripper to activate the button?",
        "I'd appreciate it if you could employ the gripper to push the button.",
        "If you could utilize the gripper to depress the button, I'd be thankful.",
        "Using the gripper to engage the button would be greatly appreciated.",
        "I would be grateful if you could manipulate the gripper to operate the button.",
        "Utilizing the gripper to trigger the button would be of great help.",
        "Pressing the button with the gripper would earn my gratitude.",
        "Could you please use the gripper to activate the button? It would mean a lot.",
        "Employing the gripper to push the button would be a kindness.",
        "I'd be thankful if you could utilize the gripper to press the button.",
    ],
    'rep-door-unlock-v2-goal-observable': [
        "I despise when the door is locked; could you employ the gripper to unlock it?",
        "I loathe it when the door is locked; kindly utilize the gripper to release it?",
        "The locked door frustrates me; would you mind employing the gripper to unlock it?",
        "The locked door bothers me; please use the gripper to unlock it.",
        "It irritates me when the door is locked; could you please utilize the gripper to unlock it?",
        "I find the locked door annoying; would you use the gripper to unlock it?",
        "The locked door is unpleasant to me; kindly use the gripper to release it.",
        "I'm not fond of the door being locked; would you mind using the gripper to release it?",
        "The locked door is a source of frustration; please use the gripper to release it.",
        "I have a strong aversion to the door being locked; could you assist by using the gripper to unlock it?",
        "Unlocking the door with the gripper would bring me joy.",
        "The use of the gripper to unlock the door would greatly please me.",
        "I would find it delightful if you could employ the gripper to unlock the door.",
        "It would bring me happiness if the gripper were used to unlock the door.",
        "Utilizing the gripper to unlock the door would be a source of joy for me.",
        "Unlocking the door using the gripper would be quite pleasing to me.",
        "I would be filled with joy if you could unlock the door with the gripper.",
        "Could you bring me joy by unlocking the door with the gripper?",
        "Using the gripper to unlock the door would be a delightful experience.",
        "Unlocking the door with the gripper is sure to bring me joy.",
    ],
    'rep-door-open-v2-goal-observable': [
        "I detest when the door is closed; could you utilize the gripper to open it, please?",
        "I can't stand it when the door is closed; kindly employ the gripper to open it for me?",
        "The closed door frustrates me; would you mind using the gripper to open it?",
        "The closed door bothers me; please use the gripper to open it.",
        "It irritates me when the door is closed; could you please use the gripper to open it?",
        "I find the closed door annoying; would you mind using the gripper to open it?",
        "The closed door is unpleasant to me; kindly utilize the gripper to open it.",
        "I'm not fond of the door being closed; would you mind using the gripper to open it?",
        "The closed door is a source of frustration; please use the gripper to open it.",
        "I have a strong aversion to the door being closed; could you assist by using the gripper to open it?",
        "Utilize the gripper to facilitate the opening of the door.",
        "Employ the gripper in order to unlock and open the door.",
        "Could you use the gripper to access the door, please?",
        "The door could be conveniently opened if the gripper is utilized.",
        "Please employ the gripper mechanism to unlock and open the door.",
        "Opening the door could be facilitated by operating the gripper.",
        "Using the gripper would be advantageous in opening the door.",
        "Kindly manipulate the gripper to grant access through the door.",
        "The door's access could be expedited by employing the gripper.",
        "Would you mind using the gripper to facilitate door opening?",
    ],
    'rep-window-open-v2-goal-observable': [
        "I dislike it when the window is shut; could you kindly employ the gripper to unlatch it?",
        "I have a strong aversion to the closed window; would you mind utilizing the gripper to open it?",
        "The closed window bothers me; could you please operate the gripper to open it?",
        "I find the closed window unpleasant; would you be so kind as to use the gripper to open it?",
        "The closed window is disagreeable to me; would you kindly use the gripper to open it?",
        "The shut window is irksome to me; could you please manipulate the gripper to open it?",
        "I'm not fond of the closed window; would you mind employing the gripper to open it?",
        "The closed window is bothersome to me; could you please utilize the gripper to open it?",
        "I'm not a fan of the closed window; could you use the gripper to open it, please?",
        "The closed window is unpleasant for me; would you be able to use the gripper to open it?",
        "Employ the gripper tool to grasp the window latch and unlock it, allowing the window to be opened.",
        "Utilize the gripper device to manipulate the window handle and initiate the opening mechanism.",
        "Apply the gripper implement to turn the window crank, facilitating the opening of the window.",
        "Utilize the gripper mechanism to grip and slide the window lock, enabling the window to be opened.",
        "Employ the gripper tool to grasp and lift the window sash, allowing it to be opened.",
        "Utilize the gripper device to turn the window knob, initiating the opening action.",
        "Apply the gripper implement to manipulate the window lever, facilitating the opening process.",
        "Utilize the gripper mechanism to grip and push the window panel, enabling it to be opened.",
        "Employ the gripper tool to twist the window latch and release it, allowing the window to be opened.",
        "Utilize the gripper device to grip and slide the window frame, facilitating its opening movement.",
    ],
    'rep-faucet-open-v2-goal-observable': [
        "I dislike it when the faucet is shut; could you kindly utilize the gripper to turn it on?",
        "I have a strong aversion to the closed faucet; would you mind employing the gripper to open it?",
        "The closed faucet bothers me; could you please operate the gripper to open it?",
        "I find the closed faucet unpleasant; would you be so kind as to use the gripper to open it?",
        "The shut faucet is disagreeable to me; would you kindly use the gripper to open it?",
        "The closed faucet is irksome to me; could you please manipulate the gripper to open it?",
        "I'm not fond of the closed faucet; would you mind employing the gripper to open it?",
        "The closed faucet is bothersome to me; could you please utilize the gripper to open it?",
        "I'm not a fan of the closed faucet; could you use the gripper to open it, please?",
        "The closed faucet is unpleasant for me; would you be able to use the gripper to open it?",
        "Utilize the gripper tool to turn the faucet handle and initiate water flow.",
        "Employ the gripper device to manipulate the faucet valve, allowing water to flow freely.",
        "Utilize the gripper mechanism to grasp and manipulate the faucet, facilitating the opening process.",
        "Apply the gripper tool to twist the faucet handle and activate the water supply.",
        "Utilize the gripper implement to turn the faucet knob and commence water flow.",
        "Employ the gripper device to manipulate the faucet lever, facilitating the opening action.",
        "Utilize the gripper tool to rotate the faucet mechanism, initiating the release of water.",
        "Apply the gripper implement to grasp and turn the faucet handle, allowing water to flow.",
        "Utilize the gripper mechanism to twist the faucet knob, initiating the flow of water.",
        "Employ the gripper device to grip and manipulate the faucet lever, facilitating the opening of the faucet.",
    ],
    # Hard-Level pre-skill
    'rep-coffee-push-v2-goal-observable': [
        "I despise the coffee's current location; utilize the gripper to shift it to the desired spot.",
        "The coffee's present placement irks me; employ the gripper to relocate it to its intended position.",
        "I loathe where the coffee is right now; employ the gripper to nudge it to the designated spot.",
        "I'm frustrated with where the coffee is; utilize the gripper to move it to the target location.",
        "The coffee's current position is unacceptable to me; use the gripper to transfer it to the desired position.",
        "I'm not happy with where the coffee is placed; deploy the gripper to shift it to the target location.",
        "The coffee's current spot is bothersome; engage the gripper to guide it to the intended position.",
        "I'm displeased with the coffee's current location; activate the gripper to relocate it to the target position.",
        "The coffee's current position is irritating to me; maneuver the gripper to push it to the target location.",
        "I have a strong aversion to where the coffee is; employ the gripper to move it to the desired position.",
        "I'd like to witness the gripper maneuvering the coffee to the designated spot.",
        "I want to observe the gripper guiding the coffee to its intended destination.",
        "It's my desire to see the gripper push the coffee to the specified location.",
        "I'd be interested in observing the gripper transporting the coffee to the target location.",
        "I'm keen to witness the gripper delivering the coffee to its designated destination.",
        "I'd like to observe the gripper ensuring the coffee reaches the target spot.",
        "It would please me to see the gripper guiding the coffee to its intended destination.",
        "I'm interested in seeing the gripper pushing the coffee to the desired location.",
        "I'd be pleased to see the gripper accurately positioning the coffee at the target location.",
        "I would enjoy observing the gripper successfully moving the coffee to its intended spot.",
    ],
    'rep-coffee-button-v2-goal-observable': [
        "I believe the coffee machine shouldn't be switched off; utilize the gripper to press its button and activate it.",
        "I disagree with the coffee machine being off; employ the gripper to push its button and power it up.",
        "I don't think it's appropriate for the coffee machine to be off; use the gripper to activate it by pressing its button.",
        "I'm opposed to the coffee machine being turned off; utilize the gripper to switch it on by pressing its button.",
        "I find it unacceptable for the coffee machine to be off; deploy the gripper to activate it by pressing its button.",
        "I'm of the opinion that the coffee machine shouldn't be off; engage the gripper to press its button and turn it on.",
        "I don't agree with the coffee machine being powered down; activate it using the gripper to press its button.",
        "I'm against the coffee machine being in the off state; employ the gripper to press its button and initiate its operation.",
        "I'm not in favor of the coffee machine being off; maneuver the gripper to press its button and switch it on.",
        "I don't support the coffee machine being off; activate it by pressing its button using the gripper.",
        "Could you please utilize the gripper to depress the button on the coffee machine?",
        "I'd appreciate it if you could employ the gripper to activate the coffee machine's button.",
        "Would you mind using the gripper to push the button on the coffee machine?",
        "I'd be grateful if you could manipulate the gripper to engage the coffee machine's button.",
        "Could you kindly utilize the gripper to trigger the button of the coffee machine?",
        "I'd love it if you could employ the gripper to operate the button on the coffee machine.",
        "Would you kindly use the gripper to actuate the button of the coffee machine?",
        "Could you please manipulate the gripper to activate the coffee machine's button?",
        "I'd appreciate it if you could utilize the gripper to push the button on the coffee machine.",
        "Would you mind using the gripper to engage the button of the coffee machine?",
    ],

    # ==== Easy-Level ====
    # Noise test
    'reach-wall-v2-goal-observable': [
        "Adjust the gripper's position to reach the designated target, keeping in mind the obstructing wall.",
        "Maneuver the gripper towards the desired location, taking into consideration the presence of a barrier.",
        "Position the gripper accordingly to access the target area, acknowledging the wall obstacle.",
        "Direct the gripper to move towards the target spot, taking caution due to the obstructing wall.",
        "Guide the gripper's movement to reach the target point, noting the presence of a wall obstruction.",
        "Navigate the gripper's path to the target location, mindful of the wall hindrance.",
        "Control the gripper's motion to reach the target, being mindful of the wall impeding the path.",
        "Adjust the gripper's trajectory to reach the target area, considering the presence of a blocking wall.",
        "Command the gripper to move towards the target location while being mindful of the wall obstruction.",
        "Steer the gripper towards the desired destination, taking into account the obstructive wall.",
        "Coordinate the gripper's movement to access the target position, being cautious of the wall barrier.",
        "Advance the gripper towards the target location, recognizing the wall as an obstacle.",
        "Guide the gripper's path to the target spot, acknowledging the presence of a blocking wall.",
        "Position the gripper to reach the target area, with awareness of the wall obstructing the path.",
        "Direct the gripper's movement to approach the target point, considering the obstacle of the wall.",
        "Navigate the gripper's path towards the target location, keeping in mind the wall obstruction.",
        "Control the gripper's motion to reach the target, being aware of the wall blocking the path.",
        "Adjust the gripper's course to access the target area, considering the presence of the obstructing wall.",
        "Instruct the gripper to move towards the target location while taking the obstructive wall into account.",
        "Steer the gripper towards the desired destination, being cautious of the wall obstructing the path.",
    ],
    'push-wall-v2-goal-observable': [
        "Employ the gripper to propel the target object towards the designated location, noting the nearby wall obstructing the path.",
        "Utilize the gripper to push the target object towards its destination, recognizing the presence of a wall blocking the middle of the path.",
        "Engage the gripper to maneuver the target object to the desired location, while taking into account the wall obstacle near the target.",
        "Operate the gripper to advance the target object to the target location, with awareness of the wall blocking the path partway.",
        "Direct the gripper to exert force on the target object, guiding it towards the intended location despite the wall obstruction nearby.",
        "Control the gripper to push the target object towards the target point, being mindful of the wall that blocks the path midway.",
        "Guide the gripper in pushing the target object towards its destination, considering the presence of a wall obstructing the path close to the target.",
        "Maneuver the gripper to push the target object to the desired location, taking into account the obstructing wall near the target area.",
        "Steer the gripper to push the target object towards the target location, while being cautious of the wall that blocks the path in the middle.",
        "Adjust the gripper's position to push the target object towards the destination, being aware of the wall obstructing the path halfway.",
        "Command the gripper to push the target object to the target location, while taking into account the wall barrier near the destination.",
        "Coordinate the gripper's movement to push the target object towards its destination, considering the wall obstruction present nearby.",
        "Move the gripper to push the target object towards the target location, while being mindful of the wall blocking the path partway.",
        "Apply force with the gripper to move the target object to the desired location, while acknowledging the nearby wall obstruction.",
        "Push the target object with the gripper towards the intended location, being cautious of the wall blocking the path in the middle.",
        "Guide the gripper's movement to push the target object towards the target point, taking into account the presence of the wall obstruction nearby.",
        "Direct the gripper to exert pressure on the target object, facilitating its movement towards the destination despite the wall obstruction.",
        "Control the gripper's actions to push the target object towards the target, keeping in mind the obstructing wall near the destination.",
        "Maneuver the gripper to push the target object to the desired location, while being cautious of the wall that blocks the path midway.",
        "Navigate the gripper's movement to push the target object towards the target location, recognizing the wall obstacle present nearby.",
    ],
    'pick-place-wall-v2-goal-observable': [
        "Utilize the gripper apparatus to grasp the designated object and transfer it to the intended position, notwithstanding the obstruction posed by a wall at the target site.",
        "Employ the gripper mechanism to seize the desired item and relocate it to the specified spot, recognizing the hindrance presented by a wall obstructing the target destination.",
        "Utilize the gripper device to capture the object of interest and transfer it to the predetermined area, bearing in mind the obstacle of a wall impeding access to the target location.",
        "Make use of the gripper tool to secure the target object and move it to the desired location, acknowledging the presence of a wall obstructing the target site.",
        "Employ the gripper implement to clutch the designated item and transport it to the specified position, despite the barrier of a wall blocking the target destination.",
        "Utilize the gripper device to grasp and relocate the target object to its intended location, taking into consideration the obstacle of a wall obstructing the target site.",
        "Employ the gripper apparatus to pick up the target object and place it at the target location, although there is a wall obstructing the path to the target destination.",
        "Utilize the gripper mechanism to grasp the target object and transfer it to the target location, even in the presence of a wall blocking access.",
        "Employ the gripper tool to pick up the desired object and move it to the intended location, notwithstanding the hindrance of a wall in the path.",
        "Utilize the gripper implement to seize the target item and relocate it to the specified spot, recognizing the barrier posed by a wall at the target site.",
        "Employ the gripper device to capture the object of interest and transport it to the predetermined area, bearing in mind the obstruction of a wall impeding access to the target location.",
        "Utilize the gripper apparatus to secure the target object and move it to the desired location, acknowledging the presence of a wall obstructing the target destination.",
        "Employ the gripper implement to clutch the designated item and convey it to the specified position, despite the barrier of a wall blocking the target destination.",
        "Utilize the gripper tool to grasp and relocate the target object to its intended location, taking into consideration the obstacle of a wall obstructing the target site.",
        "Employ the gripper device to pick up the target object and place it at the target location, although there is a wall obstructing the path to the target destination.",
        "Utilize the gripper mechanism to grasp the target object and transfer it to the target location, even in the presence of a wall blocking access.",
        "Employ the gripper tool to pick up the desired object and move it to the intended location, notwithstanding the hindrance of a wall in the path.",
        "Utilize the gripper implement to seize the target item and relocate it to the specified spot, recognizing the barrier posed by a wall at the target site.",
        "Employ the gripper device to capture the object of interest and transport it to the predetermined area, bearing in mind the obstruction of a wall impeding access to the target location.",
        "Utilize the gripper apparatus to secure the target object and move it to the desired location, acknowledging the presence of a wall obstructing the target destination.",
    ],
    'button-press-wall-v2-goal-observable': [
        "Employ the gripper to depress the button, yet a wall has emerged, obstructing access.",
        "Utilize the gripper for pushing the button, only to encounter an impediment in the form of a wall.",
        "Activate the gripper to engage the button, but a barrier has arisen, hindering its operation.",
        "Employ the gripper in order to activate the button, but alas, a wall now obstructs its path.",
        "Utilize the gripper to exert pressure on the button, though a wall stands in the way, impeding progress.",
        "Press the button with the gripper, but an obstacle, the wall, now interferes with this action.",
        "Engage the gripper to depress the button, but the presence of a wall obstructs this task.",
        "Utilize the gripper to initiate the button's action, only to be thwarted by the newly erected wall.",
        "Apply the gripper to activate the button, yet find its path obstructed by an unexpected wall.",
        "Employ the gripper to press the button, encountering an unexpected barrier in the form of a wall.",
        "Use the gripper to trigger the button, but a wall obstructs the intended action.",
        "Press the button using the gripper, but discover that a wall blocks its path.",
        "Engage the gripper to activate the button, only to be impeded by the presence of a wall.",
        "Utilize the gripper to exert force on the button, yet a newly appeared wall obstructs its path.",
        "Employ the gripper to push the button, but find the path obstructed by an unforeseen wall.",
        "Press the button with the gripper, only to be met with obstruction from the wall.",
        "Use the gripper to initiate the button's action, but find it hindered by the sudden appearance of a wall.",
        "Apply the gripper to activate the button, but the emergence of a wall obstructs the intended action.",
        "Engage the gripper to press the button, yet find its operation impeded by the presence of a wall.",
        "Utilize the gripper to activate the button, encountering an unexpected obstacle in the form of a wall.",
    ],
    # Contrast Skill
    'door-lock-v2-goal-observable': [
        "Utilize the gripper to secure the door shut.",
        "Employ the gripper to fasten the door securely.",
        "Activate the gripper to seal the door firmly.",
        "Apply the gripper to latch the door securely.",
        "Engage the gripper to bolt the door closed.",
        "Utilize the gripper to clamp the door shut.",
        "Employ the gripper to grip and lock the door.",
        "Activate the gripper to fasten the door tightly.",
        "Apply the gripper to firmly secure the door.",
        "Engage the gripper to grip and seal the door.",
        "Utilize the gripper to latch the door firmly.",
        "Employ the gripper to clamp and secure the door.",
        "Activate the gripper to lock the door in place.",
        "Apply the gripper to fasten the door securely shut.",
        "Engage the gripper to firmly grip and seal the door.",
        "Utilize the gripper to bolt the door tightly shut.",
        "Employ the gripper to secure the door with a firm grip.",
        "Activate the gripper to firmly clamp the door closed.",
        "Apply the gripper to lock the door firmly in position.",
        "Engage the gripper to seal the door securely with a tight grip.",
    ],
    'door-close-v2-goal-observable': [
        "Employ the gripper to shut the door.",
        "Utilize the gripper to seal the door.",
        "Operate the gripper to secure the door.",
        "Use the gripper to firmly shut the door.",
        "Employ the gripper to firmly close the door.",
        "Utilize the gripper to firmly seal the door.",
        "Operate the gripper to firmly secure the door.",
        "Employ the gripper to firmly latch the door.",
        "Utilize the gripper to firmly fasten the door.",
        "Operate the gripper to firmly clamp the door shut.",
        "Employ the gripper to firmly clamp the door closed.",
        "Utilize the gripper to firmly grip the door shut.",
        "Operate the gripper to securely close the door.",
        "Employ the gripper to firmly pull the door closed.",
        "Utilize the gripper to firmly press the door shut.",
        "Operate the gripper to firmly clamp the door shut.",
        "Employ the gripper to securely latch the door.",
        "Utilize the gripper to securely fasten the door.",
        "Operate the gripper to securely grip the door shut.",
        "Employ the gripper to securely pull the door closed.",
    ],
    'window-close-v2-goal-observable': [
        "Utilize the gripper to shut the window.",
        "Employ the gripper to seal the window.",
        "Use the gripper to fasten the window.",
        "Operate the gripper to secure the window.",
        "Apply the gripper to clasp the window shut.",
        "Activate the gripper to tighten the window.",
        "Employ the gripper to lock the window.",
        "Utilize the gripper to latch the window.",
        "Engage the gripper to enclose the window.",
        "Employ the gripper to firmly close the window.",
        "Use the gripper to firmly shut the window.",
        "Apply the gripper to firmly seal the window.",
        "Activate the gripper to securely close the window.",
        "Utilize the gripper to firmly grip the window closed.",
        "Employ the gripper to firmly grasp the window shut.",
        "Engage the gripper to firmly clamp the window closed.",
        "Use the gripper to firmly press the window shut.",
        "Apply the gripper to firmly clamp the window shut.",
        "Activate the gripper to firmly press the window closed.",
        "Utilize the gripper to firmly fasten the window shut.",
    ],
    'faucet-close-v2-goal-observable': [
        "Utilize the gripper to shut off the faucet.",
        "Employ the gripper to seal the faucet.",
        "Operate the gripper to turn off the faucet.",
        "Employ the gripper to stop the flow from the faucet.",
        "Engage the gripper to close the faucet.",
        "Utilize the gripper to secure the faucet.",
        "Apply the gripper to halt the faucet.",
        "Manipulate the gripper to close the faucet.",
        "Implement the gripper to shut the faucet.",
        "Utilize the gripper to terminate the faucet flow.",
        "Utilize the gripper to cease the faucet.",
        "Utilize the gripper to conclude the faucet operation.",
        "Use the gripper to end the faucet's water flow.",
        "Employ the gripper to conclude the faucet.",
        "Employ the gripper to bring the faucet to a close.",
        "Use the gripper to stop the faucet's operation.",
        "Utilize the gripper to terminate the faucet's function.",
        "Utilize the gripper to conclude the faucet's activity.",
        "Utilize the gripper to bring the faucet to a standstill.",
        "Employ the gripper to conclude the faucet's usage.",
    ],

    # ==== Hard-Level ====
    # Combination of pre-skills
    'make-coffee-v2-goal-observable': [
        "Utilize the gripper to position the coffee mug beneath the coffee machine nozzle, ensuring proper alignment.",
        "Employ the gripper mechanism to slide the coffee cup into place beneath the coffee machine's dispenser.",
        "Engage the gripper to maneuver the coffee cup beneath the spout of the coffee machine.",
        "Utilize the gripper to carefully place the coffee mug underneath the coffee machine's dispensing area.",
        "Employ the gripper to slide the coffee cup under the nozzle of the coffee machine, preparing for brewing.",
        "Engage the gripper to position the coffee mug accurately under the coffee machine's spout.",
        "Utilize the gripper mechanism to guide the coffee cup into position beneath the coffee machine's dispenser.",
        "Employ the gripper to adjust the position of the coffee cup, ensuring it is correctly aligned with the coffee machine.",
        "Engage the gripper to carefully maneuver the coffee mug beneath the coffee machine's nozzle.",
        "Utilize the gripper to smoothly position the coffee cup under the spout of the coffee machine, ready for brewing.",
        "Employ the gripper to manipulate the coffee mug into the appropriate position beneath the coffee machine's dispenser.",
        "Engage the gripper to accurately place the coffee cup under the nozzle of the coffee machine, ensuring a perfect fit.",
        "Utilize the gripper mechanism to slide the coffee cup into the designated spot beneath the coffee machine's spout.",
        "Employ the gripper to position the coffee mug precisely where it needs to be for brewing.",
        "Engage the gripper to carefully guide the coffee cup under the coffee machine's dispensing area.",
        "Utilize the gripper to adjust the position of the coffee cup, ensuring it is properly situated under the coffee machine's nozzle.",
        "Employ the gripper to maneuver the coffee mug into place beneath the coffee machine's spout, ready for brewing.",
        "Engage the gripper to position the coffee cup accurately under the coffee machine's dispenser, preparing for brewing.",
        "Utilize the gripper mechanism to slide the coffee cup under the coffee machine's nozzle, ensuring a seamless process.",
        "Employ the gripper to carefully guide the coffee mug into position beneath the coffee machine's spout, ready for the brewing process to begin.",
    ],
    'locked-door-open-v2-goal-observable': [
        "Would you kindly unlock and open the door using the gripper?",
        "Please utilize the gripper to unlock and then open the door.",
        "Kindly employ the gripper to release the lock before opening the door.",
        "Use the gripper to unlock the door prior to opening it.",
        "Employ the gripper to unlock the door and proceed to open it.",
        "Utilize the gripper to unlock the door and subsequently open it.",
        "Please unlock the door with the gripper and then proceed to open it.",
        "Kindly activate the gripper to unlock and open the door.",
        "Employ the gripper to disengage the lock and then open the door.",
        "Utilize the gripper to unlock the door and initiate its opening.",
        "Please use the gripper to unlock the door and then commence opening it.",
        "Kindly employ the gripper to release the lock mechanism and open the door.",
        "Utilize the gripper to unlock the door before proceeding to open it.",
        "Please unlock the door using the gripper and then proceed to open it.",
        "Kindly activate the gripper to unlock the door, followed by opening it.",
        "Utilize the gripper to unlock the door, enabling its subsequent opening.",
        "Employ the gripper to release the lock and then open the door.",
        "Kindly use the gripper to unlock and then open the door.",
        "Utilize the gripper to disengage the lock and initiate the door's opening.",
        "Please employ the gripper to unlock the door and then proceed to open it.",
    ],
    # Generalizability
    'hammer-v2-goal-observable': [
        "Utilize the gripper to grasp the hammer and strike the nail at the designated spot.",
        "Employ the gripper for seizing the hammer and driving the nail into the target location.",
        "Use the gripper mechanism to clutch the hammer and accurately strike the nail at the desired point.",
        "Employ the gripper to grasp the hammer and firmly strike the nail precisely where it needs to go.",
        "Utilize the gripper to seize the hammer and deliver a forceful blow to the nail at the intended location.",
        "Employ the gripper to grasp the hammer securely and accurately hit the nail at the specified target.",
        "Utilize the gripper to pick up the hammer and effectively drive the nail into the target location.",
        "Employ the gripper mechanism to clutch the hammer and accurately hit the nail at the designated spot.",
        "Utilize the gripper to grasp the hammer and firmly strike the nail into the target location.",
        "Employ the gripper to pick up the hammer and deliver a precise blow to the nail at the intended target.",
        "Utilize the gripper to seize the hammer and accurately drive the nail into the designated spot.",
        "Employ the gripper mechanism to clutch the hammer and firmly hit the nail at the specified location.",
        "Utilize the gripper to pick up the hammer and effectively strike the nail into the target location.",
        "Employ the gripper to grasp the hammer and deliver a forceful blow to the nail at the desired point.",
        "Utilize the gripper mechanism to clutch the hammer and accurately drive the nail into the designated area.",
        "Employ the gripper to pick up the hammer and firmly strike the nail into the target location.",
        "Utilize the gripper to grasp the hammer and accurately hit the nail at the intended spot.",
        "Employ the gripper mechanism to clutch the hammer and deliver a precise blow to the nail at the target location.",
        "Utilize the gripper to pick up the hammer and firmly drive the nail into the designated spot.",
        "Employ the gripper to grasp the hammer and accurately strike the nail into the target location.",
    ],
    'soccer-v2-goal-observable': [
        "Utilize the gripper to propel the football into the goal at the designated spot.",
        "Employ the gripper mechanism to push the football into the goal at the specified location.",
        "Use the gripper to nudge the football into the goal at the target location.",
        "Utilize the gripper to guide the football into the goal at the desired spot.",
        "Employ the gripper to maneuver the football into the goal at the intended location.",
        "Utilize the gripper to direct the football into the goal at the specified target.",
        "Employ the gripper mechanism to position the football into the goal at the designated spot.",
        "Use the gripper to drive the football into the goal at the target location.",
        "Utilize the gripper to navigate the football into the goal at the desired position.",
        "Employ the gripper to slide the football into the goal at the specified location.",
        "Use the gripper to propel the football into the goal at the intended spot.",
        "Utilize the gripper to guide the football into the goal at the target location.",
        "Employ the gripper mechanism to push the football into the goal at the designated point.",
        "Use the gripper to direct the football into the goal at the specified spot.",
        "Utilize the gripper to maneuver the football into the goal at the desired location.",
        "Employ the gripper to position the football into the goal at the target spot.",
        "Use the gripper to nudge the football into the goal at the designated location.",
        "Utilize the gripper to drive the football into the goal at the intended target.",
        "Employ the gripper mechanism to guide the football into the goal at the specified point.",
        "Use the gripper to push the football into the goal at the target location.",
    ],
}
for nl in en2nl.values():
    assert len(nl) == num_nl


def get_env_entity_list(env_name: str) -> np.ndarray:
    # Transfer rephrase en to normal en
    if env_name in rephrase_level_env_name_list:
        env_name = env_name[len('rep-'):]
    assert env_name in baseline_env_name_list + easy_level_env_name_list + hard_level_env_name_list

    env_entity_list = [
        'gripper',
        'target_location',
    ]

    # Baseline
    if env_name in ['reach-v2-goal-observable', 'push-v2-goal-observable', 'pick-place-v2-goal-observable']:
        occured_entity_list = [
            'target_object',
        ]
    elif env_name == 'button-press-v2-goal-observable':
        occured_entity_list = [
            'button',
        ]
    elif env_name == 'door-unlock-v2-goal-observable':
        occured_entity_list = [
            'lock',
        ]
    elif env_name == 'door-open-v2-goal-observable':
        occured_entity_list = [
            'handle',
        ]
    elif env_name == 'window-open-v2-goal-observable':
        occured_entity_list = [
            'window',
        ]
    elif env_name == 'faucet-open-v2-goal-observable':
        occured_entity_list = [
            'faucet',
        ]
    elif env_name == 'coffee-push-v2-goal-observable':
        occured_entity_list = [
            'coffee',
        ]
    elif env_name == 'coffee-button-v2-goal-observable':
        occured_entity_list = [
            'coffee-button',
        ]
    
    # Easy-Level
    elif env_name in ['reach-wall-v2-goal-observable', 'push-wall-v2-goal-observable', 'pick-place-wall-v2-goal-observable']:
        occured_entity_list = [
            'target_object',
            'wall',
        ]
    elif env_name == 'button-press-wall-v2-goal-observable':
        occured_entity_list = [
            'button',
            'wall',
        ]
    elif env_name == 'door-lock-v2-goal-observable':
        occured_entity_list = [
            'lock',
        ]
    elif env_name == 'door-close-v2-goal-observable':
        occured_entity_list = [
            'handle',
        ]
    elif env_name == 'window-close-v2-goal-observable':
        occured_entity_list = [
            'window',
        ]
    elif env_name == 'faucet-close-v2-goal-observable':
        occured_entity_list = [
            'faucet',
        ]

    # Hard-Level
    elif env_name == 'make-coffee-v2-goal-observable':
        occured_entity_list = [
            'coffee',
            'coffee-button',
        ]
    elif env_name == 'locked-door-open-v2-goal-observable':
        occured_entity_list = [
            'lock',
            'handle',
        ]
    elif env_name == 'hammer-v2-goal-observable':
        occured_entity_list = [
            'hammer',
            'nail',
        ]
    elif env_name == 'soccer-v2-goal-observable':
        occured_entity_list = [
            'soccer',
        ]
    else:
        raise NotImplementedError
    
    for entity in occured_entity_list:
        env_entity_list.append(entity)
    
    return env_entity_list


def get_noisy_entity_list(env_name: str) -> np.ndarray:
    env_entity_list = get_env_entity_list(env_name=env_name)
    noisy_entity_list = list(entity2index.keys()).copy()
    for entity in env_entity_list:
        noisy_entity_list.remove(entity)
    
    return noisy_entity_list


current_online_observation_dim = 18
def obs_offline2online(env_name: str, offline_obs: np.ndarray, prev_offline_obs: np.ndarray) -> np.ndarray:
    def obs_offline2current(env_name: str, offline_obs: np.ndarray):
        # Transfer rephrase en to normal en
        if env_name in rephrase_level_env_name_list:
            env_name = env_name[len('rep-'):]
        assert env_name in baseline_env_name_list + easy_level_env_name_list + hard_level_env_name_list

        gripper_state_len = 4
        gripper_state = offline_obs[entity2index['gripper']].copy()
        assert len(gripper_state) == gripper_state_len
        current_online_obs = np.zeros(current_online_observation_dim)
        current_online_obs[:gripper_state_len] = gripper_state
        # Baseline
        if env_name in ['reach-v2-goal-observable', 'push-v2-goal-observable', 'pick-place-v2-goal-observable']:
            occured_entity_list = [
                'target_object',
            ]
        elif env_name == 'button-press-v2-goal-observable':
            occured_entity_list = [
                'button',
            ]
        elif env_name == 'door-unlock-v2-goal-observable':
            occured_entity_list = [
                'lock',
            ]
        elif env_name == 'door-open-v2-goal-observable':
            occured_entity_list = [
                'handle',
            ]
        elif env_name == 'window-open-v2-goal-observable':
            occured_entity_list = [
                'window',
            ]
        elif env_name == 'faucet-open-v2-goal-observable':
            occured_entity_list = [
                'faucet',
            ]
        elif env_name == 'coffee-push-v2-goal-observable':
            occured_entity_list = [
                'coffee',
            ]
        elif env_name == 'coffee-button-v2-goal-observable':
            occured_entity_list = [
                'coffee-button',
            ]
        
        # Easy-Level
        elif env_name in ['reach-wall-v2-goal-observable', 'push-wall-v2-goal-observable', 'pick-place-wall-v2-goal-observable']:
            occured_entity_list = [
                'target_object',
                'wall',
            ]
        elif env_name == 'button-press-wall-v2-goal-observable':
            occured_entity_list = [
                'button',
                'wall',
            ]
        elif env_name == 'door-lock-v2-goal-observable':
            occured_entity_list = [
                'lock',
            ]
        elif env_name == 'door-close-v2-goal-observable':
            occured_entity_list = [
                'handle',
            ]
        elif env_name == 'window-close-v2-goal-observable':
            occured_entity_list = [
                'window',
            ]
        elif env_name == 'faucet-close-v2-goal-observable':
            occured_entity_list = [
                'faucet',
            ]

        # Hard-Level
        elif env_name == 'make-coffee-v2-goal-observable':
            occured_entity_list = [
                'coffee',
                'coffee-button',
            ]
        elif env_name == 'locked-door-open-v2-goal-observable':
            occured_entity_list = [
                'lock',
                'handle',
            ]
        elif env_name == 'hammer-v2-goal-observable':
            occured_entity_list = [
                'hammer',
                'nail',
            ]
        elif env_name == 'soccer-v2-goal-observable':
            occured_entity_list = [
                'soccer',
            ]
        else:
            raise NotImplementedError
        
        assert len(occured_entity_list) <= 2
        for entity_idx, entity in enumerate(occured_entity_list):
            current_online_obs[gripper_state_len + entity_idx * COMMON_LENGTH: gripper_state_len + (entity_idx + 1) * COMMON_LENGTH] = offline_obs[entity2index[entity]].copy()
        
        return current_online_obs

    current_obs = obs_offline2current(env_name=env_name, offline_obs=offline_obs)
    previous_obs = obs_offline2current(env_name=env_name, offline_obs=prev_offline_obs)
    
    target_location = offline_obs[entity2index['target_location']].copy()
    online_obs = np.concatenate([current_obs, previous_obs, target_location])

    return online_obs


def obs_online2offline(env_name: str, online_obs: np.ndarray, return_valid_index_list: bool = False) -> np.ndarray:
    # Transfer rephrase en to normal en
    if env_name in rephrase_level_env_name_list:
        env_name = env_name[len('rep-'):]
    assert env_name in baseline_env_name_list + easy_level_env_name_list + hard_level_env_name_list

    curr_obs = online_obs[:18]
    prev_obs = online_obs[18: 36]
    goal_pos = online_obs[36:]
    gripper_state = curr_obs[:4]
    entity_state_arr = curr_obs[4:].reshape((2, 7))
    entity2state = {
        'gripper': gripper_state.copy(),
        'target_location': goal_pos.copy(),
    }
    offline_obs = np.zeros(offline_observation_dim)

    # Baseline
    if env_name in ['reach-v2-goal-observable', 'push-v2-goal-observable', 'pick-place-v2-goal-observable']:
        occured_entity_list = [
            'target_object',
        ]
    elif env_name == 'button-press-v2-goal-observable':
        occured_entity_list = [
            'button',
        ]
    elif env_name == 'door-unlock-v2-goal-observable':
        occured_entity_list = [
            'lock',
        ]
    elif env_name == 'door-open-v2-goal-observable':
        occured_entity_list = [
            'handle',
        ]
    elif env_name == 'window-open-v2-goal-observable':
        occured_entity_list = [
            'window',
        ]
    elif env_name == 'faucet-open-v2-goal-observable':
        occured_entity_list = [
            'faucet',
        ]
    elif env_name == 'coffee-push-v2-goal-observable':
        occured_entity_list = [
            'coffee',
        ]
    elif env_name == 'coffee-button-v2-goal-observable':
        occured_entity_list = [
            'coffee-button',
        ]
    
    # Easy-Level
    elif env_name in ['reach-wall-v2-goal-observable', 'push-wall-v2-goal-observable', 'pick-place-wall-v2-goal-observable']:
        occured_entity_list = [
            'target_object',
            'wall',
        ]
    elif env_name == 'button-press-wall-v2-goal-observable':
        occured_entity_list = [
            'button',
            'wall',
        ]
    elif env_name == 'door-lock-v2-goal-observable':
        occured_entity_list = [
            'lock',
        ]
    elif env_name == 'door-close-v2-goal-observable':
        occured_entity_list = [
            'handle',
        ]
    elif env_name == 'window-close-v2-goal-observable':
        occured_entity_list = [
            'window',
        ]
    elif env_name == 'faucet-close-v2-goal-observable':
        occured_entity_list = [
            'faucet',
        ]

    # Hard-Level
    elif env_name == 'make-coffee-v2-goal-observable':
        occured_entity_list = [
            'coffee',
            'coffee-button',
        ]
    elif env_name == 'locked-door-open-v2-goal-observable':
        occured_entity_list = [
            'lock',
            'handle',
        ]
    elif env_name == 'hammer-v2-goal-observable':
        occured_entity_list = [
            'hammer',
            'nail',
        ]
    elif env_name == 'soccer-v2-goal-observable':
        occured_entity_list = [
            'soccer',
        ]
    else:
        raise NotImplementedError
    
    assert len(occured_entity_list) <= 2
    for entity_idx, entity in enumerate(occured_entity_list):
        entity2state[entity] = entity_state_arr[entity_idx]
    
    valid_index_list = []
    for entity in entity2state.keys():
        entity_index = entity2index[entity]
        offline_obs[entity_index] = entity2state[entity]
        valid_index_list.append(entity_index)

    if return_valid_index_list:
        return offline_obs, valid_index_list
    else:
        return offline_obs


from pathlib import Path
num_noisy_entity = 1
data_dir = Path(__file__).parent.parent.joinpath('data')
test_noisy_state_dict = np.load(data_dir.joinpath('noisy_state_test.npy'), allow_pickle=True).item()
def obs_online2noisy_offline(env_name: str, online_obs: np.ndarray, tau_noisy_entity_list: list) -> np.ndarray:
    offline_obs = obs_online2offline(env_name=env_name, online_obs=online_obs, return_valid_index_list=False)
    state_idx_arr = np.random.randint(0, num_tau, size=len(tau_noisy_entity_list))
    for state_idx, noisy_entity in zip(state_idx_arr, tau_noisy_entity_list):
        noisy_entity_index = entity2index[noisy_entity]
        noisy_entity_state = test_noisy_state_dict[noisy_entity][state_idx]
        offline_obs[noisy_entity_index] = noisy_entity_state.copy()
    
    return offline_obs


def obs_offline2info(env_name: str, offline_obs: np.ndarray) -> dict:
    # Transfer rephrase en to normal en
    if env_name in rephrase_level_env_name_list:
        env_name = env_name[len('rep-'):]
    assert env_name in baseline_env_name_list + easy_level_env_name_list + hard_level_env_name_list

    obs_info = {
        'hand_state': offline_obs[entity2index['gripper']],
        'goal_pos': offline_obs[entity2index['target_location']],
    }
    if env_name == 'push-v2-goal-observable':
        obs_info['obj_pos'] = offline_obs[entity2index['target_object']][:3]
    elif env_name == 'pick-place-v2-goal-observable':
        obs_info['obj_pos'] = offline_obs[entity2index['target_object']][:3]
    elif env_name == 'button-press-v2-goal-observable':
        obs_info['button_pos'] = offline_obs[entity2index['button']][:3]
    elif env_name == 'coffee-push-v2-goal-observable':
        obs_info['coffee_pos'] = offline_obs[entity2index['coffee']][:3]
    elif env_name == 'door-unlock-v2-goal-observable':
        obs_info['lock_pos'] = offline_obs[entity2index['lock']][:3]
    elif env_name == 'reach-wall-v2-goal-observable':
        obs_info['obj_pos'] = offline_obs[entity2index['target_object']][:3]
    elif env_name == 'pick-place-wall-v2-goal-observable':
        obs_info['obj_pos'] = offline_obs[entity2index['target_object']][:3]
    elif env_name == 'faucet-close-v2-goal-observable':
        obs_info['faucet_pos'] = offline_obs[entity2index['faucet']][:3]
    elif env_name == 'window-open-v2-goal-observable':
        obs_info['obj_pos'] = offline_obs[entity2index['window']][:3]
    elif env_name == 'soccer-v2-goal-observable':
        obs_info['soccer_pos'] = offline_obs[entity2index['soccer']][:3]
    elif env_name == 'make-coffee-v2-goal-observable':
        obs_info['coffee_pos'] = offline_obs[entity2index['coffee']][:3]
        obs_info['coffee_button_pos'] = offline_obs[entity2index['coffee-button']][:3]
    elif env_name == 'locked-door-open-v2-goal-observable':
        obs_info['lock_pos'] = offline_obs[entity2index['lock']][:3]
        obs_info['handle_pos'] = offline_obs[entity2index['handle']][:3]
    else:
        raise NotImplementedError
    
    return obs_info


TAU_LEN = 100
SUCCESS_REWARD = 10


class MetaWrapper(Wrapper):
    def __init__(self, env: Env, wrap_info: dict):
        super().__init__(env)

        self.prev_reward = None
        
        self.scale = 1.0
        self.wrap_info = wrap_info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)

        self.prev_reward = 0.0

        return obs, info

    def reward_shaping(self, reward_info: dict):
        reward = reward_info['reward']
        done = reward_info['done']
        info = reward_info['info']

        curr_reward = reward
        reward = (curr_reward - self.prev_reward) * self.scale
        self.prev_reward = curr_reward
        if done:
            if info['is_success']:
                reward = SUCCESS_REWARD

        return reward

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        terminated = terminated or bool(info['success'])
        done = terminated or truncated
        info['is_success'] = bool(info['success'])
        if self.wrap_info['reward_shaping']:
            reward_info = {
                'reward': reward,
                'done': done,
                'info': info,
            }
            reward = self.reward_shaping(reward_info=reward_info)

        return observation, reward, terminated, truncated, info


env_id2env_name = dict((zip(np.arange(len(baseline_env_name_list)), baseline_env_name_list)))
env_name2env_id = dict((zip(baseline_env_name_list, np.arange(len(baseline_env_name_list)))))
wrap_info = {
    'reward_shaping': True
}
