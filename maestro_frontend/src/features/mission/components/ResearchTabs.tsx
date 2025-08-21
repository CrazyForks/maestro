import React, { useEffect, useState } from 'react'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '../../../components/ui/tabs'
import { useMissionStore } from '../store'
import { PlanTab } from './PlanTab'
import { NotesTab } from './NotesTab'
import { DraftTab } from './DraftTab'
import { AgentsTab } from './AgentsTab'
import { GoalPadTab } from './GoalPadTab'
import { ScratchpadTab } from './ScratchpadTab'
import { ThoughtPadTab } from './ThoughtPadTab'
import { FileText, BookOpen, Edit3, Bot, } from 'lucide-react'
import { apiClient } from '../../../config/api'
import type { MissionContext } from '../types'
interface ResearchTabsProps {
  missionId: string
  isWebSocketConnected?: boolean
  hasMoreLogs?: boolean
  onLoadMoreLogs?: () => void
  onLoadAllLogs?: () => void
  isLoadingMoreLogs?: boolean
  totalLogsCount?: number
}

export const ResearchTabs: React.FC<ResearchTabsProps> = ({ 
  missionId, 
  hasMoreLogs,
  onLoadMoreLogs,
  onLoadAllLogs,
  isLoadingMoreLogs,
  totalLogsCount
}) => {
  const { activeTab, setActiveTab, ensureMissionInStore, missionContexts, fetchMissionContext } = useMissionStore()
  const [activePlanTab, setActivePlanTab] = useState('outline')
  
  // Get mission context from store - will be updated via WebSocket
  const missionContext = missionContexts[missionId] || null

  // Removed duplicate WebSocket connection - ResearchPanel already handles mission WebSocket

  useEffect(() => {
    if (missionId) {
      ensureMissionInStore(missionId)
      // Only fetch context if we don't have it in the store yet
      if (!missionContext) {
        fetchMissionContext(missionId)
      }
    }
  }, [missionId, ensureMissionInStore, fetchMissionContext, missionContext])

  // Removed WebSocket message handling - ResearchPanel handles mission WebSocket updates

  return (
    <div className="w-full h-full flex flex-col">
      <Tabs value={activeTab} onValueChange={(value) => setActiveTab(value as any)} className="h-full flex flex-col">
        <TabsList className="grid w-full grid-cols-4 flex-shrink-0 bg-secondary">
          <TabsTrigger value="plan" className="flex items-center gap-2">
            <FileText className="h-4 w-4" />
            <span className="hidden sm:inline">Plan</span>
          </TabsTrigger>
          <TabsTrigger value="agents" className="flex items-center gap-2">
            <Bot className="h-4 w-4" />
            <span className="hidden sm:inline">Agents</span>
          </TabsTrigger>
          <TabsTrigger value="notes" className="flex items-center gap-2">
            <BookOpen className="h-4 w-4" />
            <span className="hidden sm:inline">Notes</span>
          </TabsTrigger>
          <TabsTrigger value="draft" className="flex items-center gap-2">
            <Edit3 className="h-4 w-4" />
            <span className="hidden sm:inline">Draft</span>
          </TabsTrigger>
        </TabsList>

        <TabsContent value="plan" className="mt-2 flex-1 overflow-auto">
          <Tabs value={activePlanTab} onValueChange={setActivePlanTab} className="h-full flex flex-col">
            <TabsList className="grid w-full grid-cols-4 bg-secondary">
              <TabsTrigger value="outline">Outline</TabsTrigger>
              <TabsTrigger value="goal_pad">Goal Pad</TabsTrigger>
              <TabsTrigger value="scratchpad">Scratchpad</TabsTrigger>
              <TabsTrigger value="thought_pad">Thought Pad</TabsTrigger>
            </TabsList>
            <TabsContent value="outline" className="mt-2 flex-1 overflow-auto">
              <PlanTab missionId={missionId} />
            </TabsContent>
            <TabsContent value="goal_pad" className="mt-2 flex-1 overflow-auto">
              <GoalPadTab goals={missionContext?.goal_pad || []} />
            </TabsContent>
            <TabsContent value="scratchpad" className="mt-2 flex-1 overflow-auto">
              <ScratchpadTab scratchpad={missionContext?.agent_scratchpad || null} />
            </TabsContent>
            <TabsContent value="thought_pad" className="mt-2 flex-1 overflow-auto">
              <ThoughtPadTab thoughts={missionContext?.thought_pad || []} />
            </TabsContent>
          </Tabs>
        </TabsContent>

        <TabsContent value="agents" className="mt-2 flex-1 overflow-auto">
          <AgentsTab 
            missionId={missionId}
            hasMoreLogs={hasMoreLogs}
            onLoadMoreLogs={onLoadMoreLogs}
            onLoadAllLogs={onLoadAllLogs}
            isLoadingMoreLogs={isLoadingMoreLogs}
            totalLogsCount={totalLogsCount}
          />
        </TabsContent>

        <TabsContent value="notes" className="mt-2 flex-1 overflow-auto">
          <NotesTab missionId={missionId} />
        </TabsContent>

        <TabsContent value="draft" className="mt-2 flex-1 overflow-auto">
          <DraftTab missionId={missionId} />
        </TabsContent>
      </Tabs>
    </div>
  )
}
