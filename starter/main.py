"""
Customer Support AI Agent
=========================
Production-ready Customer Support AI Agent built on Amazon Bedrock AgentCore,
Strands Agents, Knowledge Base RAG, MCP Gateway, and Long-Term Memory.
"""

import argparse
import asyncio
import json
import logging
import os
import uuid
from typing import Dict

import boto3
import botocore.auth
import botocore.awsrequest
import httpx
from bedrock_agentcore.memory import MemoryClient
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.tools.code_interpreter_client import code_session
from mcp.client.streamable_http import streamable_http_client
from strands import Agent, tool
from strands.hooks import (
    AfterInvocationEvent,
    HookProvider,
    HookRegistry,
    MessageAddedEvent,
)
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from strands_tools.browser import AgentCoreBrowser


class SigV4Auth(httpx.Auth):
    """Custom HTTPX Auth class for AWS SigV4 request signing."""

    def __init__(self, credentials, region: str, service: str):
        self.credentials = credentials
        self.region = region
        self.service = service
        self.signer = botocore.auth.SigV4Auth(credentials, service, region)

    def auth_flow(self, request):
        aws_request = botocore.awsrequest.AWSRequest(
            method=request.method,
            url=str(request.url),
            headers=dict(request.headers),
            data=request.content,
        )
        self.signer.add_auth(aws_request)
        for k, v in aws_request.headers.items():
            request.headers[k] = v
        yield request


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("CSAI_Agent")

# App Initialization
app = BedrockAgentCoreApp()
os.environ["BYPASS_TOOL_CONSENT"] = "true"

# Configuration
GATEWAY_URL = "https://customersupportgateway2-jvzzxg88o1.gateway.bedrock-agentcore.us-west-2.amazonaws.com/mcp"
KB_ID = "XYFRQHIQZM"
REGION = "us-west-2"
MEMORY_ID = "CustomerSupportMemory-qtJNX4DKY7"

# Model and Clients
model_id = "global.amazon.nova-2-lite-v1:0"
model = BedrockModel(model_id=model_id)
memory_client = MemoryClient(region_name=REGION)
_bedrock_runtime = boto3.client("bedrock-agent-runtime", region_name=REGION)


def _extract_text_from_content(content) -> str:
    """Helper to extract text from string or Strands ContentBlock list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for b in content:
            if isinstance(b, dict):
                if "text" in b:
                    texts.append(b["text"])
                elif b.get("type") == "text" and "text" in b:
                    texts.append(b["text"])
            elif hasattr(b, "text"):
                texts.append(getattr(b, "text", ""))
        return " ".join(texts)
    return str(content) if content else ""


def get_namespaces(mem_client: MemoryClient, memory_id: str) -> Dict[str, str]:
    """Return a dict mapping strategy type to namespace template string."""
    try:
        strategies = mem_client.get_memory_strategies(memory_id)
        result = {}
        for strategy in strategies:
            stype = strategy.get("type")
            if "namespaceTemplates" in strategy and strategy["namespaceTemplates"]:
                result[stype] = strategy["namespaceTemplates"][0]
            elif "namespaces" in strategy and strategy["namespaces"]:
                result[stype] = strategy["namespaces"][0]
        return result
    except Exception as e:
        logger.warning(f"Failed to get memory strategies: {e}")
        return {
            "SEMANTIC": "cs_agent/{actorId}/facts",
            "USER_PREFERENCE": "cs_agent/{actorId}/preferences",
        }


class MemoryHook(HookProvider):
    """Long-term memory hook for the customer support agent."""

    def __init__(
        self,
        actor_id: str,
        session_id: str,
        memory_client: MemoryClient,
        memory_id: str,
    ):
        self.actor_id = actor_id
        self.session_id = session_id
        self.memory_client = memory_client
        self.memory_id = memory_id
        self.namespaces = get_namespaces(memory_client, memory_id)
        if not self.namespaces:
            self.namespaces = {
                "SEMANTIC": "cs_agent/{actorId}/facts",
                "USER_PREFERENCE": "cs_agent/{actorId}/preferences",
            }

    def retrieve_customer_context(self, event: MessageAddedEvent):
        """Retrieve relevant memories and prepend them to the user message."""
        if not event.agent.messages:
            return

        last_msg = event.agent.messages[-1]
        role = last_msg.get("role") if isinstance(last_msg, dict) else getattr(last_msg, "role", None)
        if role != "user":
            return

        content = last_msg.get("content") if isinstance(last_msg, dict) else getattr(last_msg, "content", None)

        if isinstance(content, list):
            for block in content:
                if (isinstance(block, dict) and "toolResult" in block) or hasattr(block, "toolResult"):
                    return

        query = _extract_text_from_content(content)
        if not query or not query.strip():
            return

        memories = []
        for stype, namespace_template in self.namespaces.items():
            namespace = namespace_template.replace("{actorId}", self.actor_id)
            try:
                results = self.memory_client.retrieve_memories(
                    memory_id=self.memory_id,
                    namespace=namespace,
                    query=query,
                    top_k=5,
                )
                for mem in results:
                    content_obj = mem.get("content", {}) if isinstance(mem, dict) else getattr(mem, "content", {})
                    txt = content_obj.get("text", "") if isinstance(content_obj, dict) else getattr(content_obj, "text", "")
                    if not txt:
                        txt = mem.get("text", "") if isinstance(mem, dict) else getattr(mem, "text", "")
                    if txt and txt not in memories:
                        memories.append(f"[{stype}] {txt}")
            except Exception as e:
                logger.warning(f"retrieve_memories failed for {namespace}: {e}")

        if memories:
            context_str = "Customer Context:\n" + "\n".join(memories) + "\n\n" + query
            if isinstance(content, list):
                updated = False
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        block["text"] = context_str
                        updated = True
                        break
                    elif hasattr(block, "text"):
                        block.text = context_str
                        updated = True
                        break
                if not updated:
                    content.insert(0, {"text": context_str})
            elif isinstance(last_msg, dict):
                last_msg["content"] = context_str
            else:
                last_msg.content = context_str

    def save_support_interaction(self, event: AfterInvocationEvent):
        """Save the completed turn to memory after the agent responds."""
        user_msg = None
        asst_msg = None

        for msg in reversed(event.agent.messages):
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)

            if role == "assistant" and not asst_msg:
                txt = _extract_text_from_content(content)
                if txt and txt.strip():
                    asst_msg = txt

            elif role == "user" and not user_msg:
                is_tool = False
                if isinstance(content, list):
                    for b in content:
                        if (isinstance(b, dict) and "toolResult" in b) or hasattr(b, "toolResult"):
                            is_tool = True
                            break
                if not is_tool:
                    txt = _extract_text_from_content(content)
                    if txt and txt.strip():
                        user_msg = txt

            if user_msg and asst_msg:
                break

        if user_msg and asst_msg:
            if "Customer Context:\n" in user_msg:
                parts = user_msg.split("\n\n", 1)
                if len(parts) > 1:
                    user_msg = parts[1]

            try:
                self.memory_client.create_event(
                    memory_id=self.memory_id,
                    actor_id=self.actor_id,
                    session_id=self.session_id,
                    messages=[(user_msg, "USER"), (asst_msg, "ASSISTANT")],
                )
            except Exception as e:
                logger.warning(f"Failed to save interaction: {e}")

    def register_hooks(self, registry: HookRegistry) -> None:  # type: ignore
        """Register memory callbacks."""
        registry.add_callback(MessageAddedEvent, self.retrieve_customer_context)
        registry.add_callback(AfterInvocationEvent, self.save_support_interaction)


@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the Amazon product catalog and support knowledge base.
    Use this for product specifications, return policies, warranty
    information, loyalty program details, and order status definitions.

    Args:
        query: The question or topic to search for

    Returns:
        Relevant information retrieved from the knowledge base
    """
    if not KB_ID or KB_ID == "<kbid>":
        return "Knowledge base not configured."

    try:
        resp = _bedrock_runtime.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": query},
        )
        results = resp.get("retrievalResults", [])
        if not results:
            return "No information found."

        chunks = [r.get("content", {}).get("text", "") for r in results]
        return "\n---\n".join(chunks)
    except Exception as e:
        return f"Error retrieving from knowledge base: {e}"


@tool
def calculate_loyalty_discount(
    loyalty_points: int,
    tier: str,
    order_total: float,
    product_category: str = "standard",
) -> str:
    """
    Calculate the loyalty discount for a customer order using the
    AgentCore Code Interpreter. Runs exact arithmetic in a secure sandbox.

    Args:
        loyalty_points: Customer's current points balance
        tier: Customer tier — Silver, Gold, or Platinum
        order_total: Order total in USD
        product_category: standard, device, or fresh

    Returns:
        Full discount breakdown and final price
    """
    code = f"""
import json

earn_rates = {{"standard": 1, "device": 2, "fresh": 5}}
tier_rates = {{"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}}

points = {loyalty_points}
tier = "{tier}"
total = {order_total}
category = "{product_category}"

points_redeemed = (points // 500) * 500
discount_from_points = points_redeemed * 0.01

if discount_from_points > total * 0.5:
    points_redeemed = int((total * 0.5) / 0.01) // 500 * 500
    discount_from_points = points_redeemed * 0.01

subtotal = total - discount_from_points
tier_discount = subtotal * tier_rates.get(tier, 0.0)
final_total = subtotal - tier_discount

points_earned = int(final_total * earn_rates.get(category, 1))
remaining_points = points - points_redeemed + points_earned

result = {{
    "points_redeemed": points_redeemed,
    "tier_discount_pct": tier_rates.get(tier, 0.0) * 100,
    "final_total": final_total,
    "remaining_points": remaining_points
}}
print(json.dumps(result))
"""

    try:
        session = code_session(REGION)
        result = session.invoke("executeCode", {"language": "python", "code": code, "clearContext": True})
        if isinstance(result, list) and len(result) > 0:
            return json.dumps(result[0])
        elif isinstance(result, dict):
            return json.dumps(result)
        else:
            return str(result)
    except Exception as e:
        tier_rates = {"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}
        tier_discount_pct = tier_rates.get(tier, 0.0)
        tier_discount = order_total * tier_discount_pct
        final_total = order_total - tier_discount
        return json.dumps({
            "points_redeemed": 0,
            "tier_discount_pct": tier_discount_pct * 100,
            "final_total": final_total,
            "remaining_points": loyalty_points,
        })


@app.entrypoint
async def invoke(payload, context=None):
    """
    Main handler called by AgentCore for every incoming request.
    """
    user_input = payload.get("prompt")
    actor_id = payload.get("customer_id", "anonymous")
    session_id = payload.get("session_id", str(uuid.uuid4()))

    if not user_input:
        return "Error: prompt is required."

    try:
        memory_hook = MemoryHook(
            actor_id=actor_id,
            session_id=session_id,
            memory_client=memory_client,
            memory_id=MEMORY_ID,
        )

        agent_core_browser = AgentCoreBrowser(region=REGION)
        tools_list = [search_knowledge_base, calculate_loyalty_discount, agent_core_browser.browser]

        session = boto3.Session(region_name=REGION)
        creds = session.get_credentials().get_frozen_credentials()
        auth = SigV4Auth(creds, REGION, "bedrock-agentcore")

        async with httpx.AsyncClient(auth=auth) as http_client:
            mcp_client = MCPClient(lambda: streamable_http_client(GATEWAY_URL, http_client=http_client))
            gateway_tools = await mcp_client.load_tools()
            tools_list.extend(gateway_tools)

            agent = Agent(
                model=model,
                tools=tools_list,
                system_prompt="You are a helpful customer support agent for an e-commerce platform.",
            )
            memory_hook.register_hooks(agent.hooks)

            response = await agent.invoke_async(user_input)

            if hasattr(response, "content"):
                if isinstance(response.content, str):
                    return response.content
                elif isinstance(response.content, list):
                    for block in response.content:
                        if isinstance(block, dict):
                            if "text" in block:
                                return block["text"]
                            elif block.get("type") == "text":
                                return block.get("text", "")
                        elif hasattr(block, "text"):
                            return getattr(block, "text", "")
            return str(response)

    except Exception as e:
        import traceback

        err_details = "".join(traceback.format_exception(e))
        logger.error(f"Error during agent invocation: {err_details}")
        return f"An error occurred: {err_details}"


def main():
    """Run one invocation from the command line for local testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=str)
    args = parser.parse_args()
    response = asyncio.run(invoke(json.loads(args.payload)))
    print(response)


if __name__ == "__main__":
    app.run()
