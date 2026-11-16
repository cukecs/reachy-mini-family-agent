# 身份识别回退 / Speaker identity fallback

## 中文

每轮交流按以下顺序决定身份：

1. 优先使用本轮声纹匹配与稳定器结果。边界声纹仍需要连续两次同人匹配；强声纹可以直接确认。
2. 声纹未确认时，尝试本轮摄像头检测到的人脸。仅当画面中检测到一张脸、特征匹配已通过、质量加权置信度至少为 0.68，才采用 `face-fallback` 身份。
3. 多人同框（包括一个认识的人和一个陌生人）、人脸不匹配或声纹候选与人脸冲突时，不按居中、面积或上一轮身份猜测。已确认的声纹始终优先，不会被旁观者的人脸覆盖。

人脸回退是一种单人可见场景下的交流假设，不是证明画面中的人正在发声；镜头外有人说话仍可能误判。不要用于安全认证。

回退结果会传给本轮模型提示、称呼、问候和对话归属，并明确标记来源，不会谎称声纹已匹配。家庭模式下按既有规则读取对应个人记忆；非家庭模式下单一模态不自动开放私人记忆。高置信度回退（至少 0.82）可用于现有的明确姓名/姓氏记忆更新，较低置信度仍保留更新限制。

回退本身不自动更新声纹库。只有已有的可靠声纹确认，或用户明确自我介绍且姓名与当前识别档案一致时，才按相应流程维护/绑定声音，避免把旁观者的脸和另一人的声音混入同一档案。

身份结果仅使用本轮帧。摄像头异常/没有脸时不会沿用历史人脸；声纹异常时仍可尝试本轮人脸。修改后重启 Agent 生效，不需要重新构建控制台 EXE。

## English

Each utterance prefers stabilized voice identity. If voice is unresolved, the Agent can fall back to a single face detected in the current frame, already accepted by the face matcher and with quality-weighted confidence >= 0.68. Multiple faces (including unknown faces), rejected matches or a conflicting pending voice candidate do not select a speaker. Confirmed voice takes precedence over visible bystanders.

The result is explicitly labelled `face-fallback` in the model context and turn attribution, and reaches greetings and profile-memory selection under the existing family/private-memory policy. This assumes the sole visible person is speaking; it is not audiovisual verification or security authentication. No previous-frame identity is carried forward.

Fallback alone does not automatically enrol/update voice templates. Explicit matching-name self-introduction can bind voice to the recognized profile instead of creating a duplicate. Restart the Agent to load the change; the desktop EXE does not need rebuilding.
