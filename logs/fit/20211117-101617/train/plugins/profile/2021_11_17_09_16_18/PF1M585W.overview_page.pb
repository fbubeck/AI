?($	???y????z1u???
h"lxz??!%u???$	?
=h?V@wg?,s?@
+Lf??@!?f?h&?2@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$ё\?C??????ׁs??A,e?X??Yf?c]?F??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?@??ǘ??q=
ףp??A?N@a???Y?U???؟?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&|a2U0??j?q?????A?G?z???Y?!??u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??????????z6??A?0?*???Y?Zd;??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&"lxz?,??0*??D??A c?ZB>??YO??e?c??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?.n?????:M???A?B?i?q??Y? ?	???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?Q???U???N@??A?l??????Yvq?-??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Y?8??m??'?W???Ae?`TR'??Yu????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&U0*????ffffff??A!?rh????YQ?|a??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	^?I+????j+????A???????Y?p=
ף??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
Έ?????<Nё\???A?A?f???Yr??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????Mb??KY?8????A????K7??Y??Pk?w??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&S??:??M?O????A)??0???Y???QI??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&-!?lV???/L?
F??As??A??Y???x?&??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&6?>W[???V}??b??AGr?????Yz6?>W[??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?C?l????46<?R??A?8EGr???Y}гY????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&%u???B>?٬???Au?V??Y?? ?rh??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	??g?????"??~j??A??e??a??Y???H??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?\?C???????1????A?"??~j??Y)\???(??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&q=
ףp??z6?>W[??A??s????Y=?U?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
h"lxz???:pΈҮ?A?H?}8??YT㥛? ??*	     ??@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??镲??!??0
K@@)?Ǻ????1FL??1?<@:Preprocessing2F
Iterator::Model????_v??!??t?hF@)jM??S??1?>D߱:@:Preprocessing2U
Iterator::Model::ParallelMapV2)??0???!?u????1@))??0???1?u????1@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??j+????!{E???K@)Έ?????1h[??!@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice$???~???!?(,E?m@)$???~???1?(,E?m@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate??x?&1??!?S?m?&@)??A?f??1q?y?;+@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*??e?c]??!G"?"?
@)??e?c]??1G"?"?
@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapŏ1w-!??!??%(V-@)??6???1?????&
@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 48.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9;	ą?c@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	??|]??,f?Y???:pΈҮ?!q=
ףp??	!       "	!       *	!       2$	??C??tw?N???H?}8??!u?V??:	!       B	!       J$	?*??ʠ????????=?U?????!f?c]?F??R	!       Z$	?*??ʠ????????=?U?????!f?c]?F??JCPU_ONLYY;	ą?c@b Y      Y@q???]???@"?
both?Your program is POTENTIALLY input-bound because 48.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?31.6431% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 