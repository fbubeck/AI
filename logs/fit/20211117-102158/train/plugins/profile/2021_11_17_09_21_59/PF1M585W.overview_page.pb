?($	H?f	????.?M?????_?L??!??&???$	?Q	Jة@?1?W?@?2[X??@!?QB0@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?? ?	???S㥛İ?A???S???Yo?ŏ1??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?m4??@??d;?O????A0*??D??Yp_?Q??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&K?46????(\????AB>?٬???Y	?c???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??:M???rh??|??A???V?/??Y?:pΈҞ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??&??????&S??A.?!??u??YtF??_??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&F??_????&?W??A???<,???YǺ?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??#?????1?*????A-???????Y/?$???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??d?`T???H?}??A?'????Y??JY?8??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&q=
ףp??}??b???A*:??H??Y??JY?8??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?3??7????ZӼ???A????9#??Y?I+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?? ?	??????(??A?@??ǘ??Y?5?;Nё?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???H??1?Zd??Aa2U0*???Y??0?*??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&W?/?'??????A/?$???Y?~j?t???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&L7?A`???$(~??k??A)\???(??Y??d?`T??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&[????<???H?}??Aё\?C???YΈ?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&g??j+?????_?L??A;pΈ????Y???߾??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&h"lxz????A?f????A??ܵ?|??Y???B?i??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&F%u???^K?=???A??o_??YJ+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&*??D?????<,Ԛ??A"??u????YZd;?O???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&w-!?l???/L?
F??AZ??ڊ???Ya??+e??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??_?L???[ A???Aq???h??Y?+e?X??*?????x?@)      @=2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatQ?|a??!?xTX^VB@)?!??u???1n??A?@@:Preprocessing2F
Iterator::Model??(???!?6??B@)<?R?!???1TÿZ?5@:Preprocessing2U
Iterator::Model::ParallelMapV2?!?uq??!JU??	-@)?!?uq??1JU??	-@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?x?&1??!?=??O@)??&S??1?;nL??"@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?A`??"??!#?g@)?A`??"??1#?g@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate	?c???!??d??*@)}гY????1?ݸ˩@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapB`??"???!W??MV?1@) o?ŏ??1?#?&?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*???Mb??!m????@)???Mb??1m????@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 53.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9???a??@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?Z%Ad??V?-L????S㥛İ?!??(\????	!       "	!       *	!       2$	"SS?????*|?4??q???h??!.?!??u??:	!       B	!       J$	???<,Ԛ??a?R?[z??5?;Nё?!o?ŏ1??R	!       Z$	???<,Ԛ??a?R?[z??5?;Nё?!o?ŏ1??JCPU_ONLYY???a??@b Y      Y@qdHB0D@"?
both?Your program is POTENTIALLY input-bound because 53.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?40.377% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 