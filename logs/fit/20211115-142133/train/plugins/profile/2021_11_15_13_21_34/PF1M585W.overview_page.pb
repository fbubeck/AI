?($	n?w?q????0??l??Z??ڊ???!c?ZB>???$	?w#?f@??GN `@܅?q??@!IM?OP?5@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$Z??ڊ????_vO??AQk?w????Y??<,Ԛ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&c?ZB>??????{????A?ݓ??Z??YM??St$??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?u???????+e???A?t?V??Y?ܵ?|У?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
ףp=
????(????A?:M???Y?4?8EG??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&V-?????e??a??A'???????Y*??Dؠ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?H.?!???RI??&???A4??@????YY?8??m??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&\???(\??vq?-??A?	h"lx??Y??ZӼ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????????z?):????A?rh??|??Y?:pΈ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&K?46??ףp=
???AHP?s???Y_?Qڛ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?Y??ڊ???m4??@??A46<?R??Y}гY????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
!?lV}??o?ŏ1??A?W?2??Ya??+e??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??3????{?/L?
??A???o_??Y?q??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Ǻ????1?Zd??A?$??C??Y46<?R??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Ԛ?????鷯???A(??y??Y
ףp=
??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?:pΈ????46<??Aŏ1w-!??Y{?G?z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&c?ZB>???+?????A?e?c]???Y	?^)ː?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????Q????(???A2??%????Y???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&tF??_??=,Ԛ???A?Fx$??Y???{????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?f??j+???,C????A?lV}???Y??+e???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??u?????2ı.n??AгY?????Y{?G?z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?&S????D???J??A????(??Y???9#J??*	?????l?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?3??7???!?i%tz?@)z6?>W[??1??nR?<@:Preprocessing2F
Iterator::ModelI.?!????!2?ީk9B@)?@??ǘ??1?z???6@:Preprocessing2U
Iterator::Model::ParallelMapV2?|a2U??!6??XQ+@)?|a2U??16??XQ+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???&??!?~!V??O@)W?/?'??1"[e^~s&@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?3??7???!?i%tz/@)8gDio???1??c??3"@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice      ??!?
???@)      ??1?
???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?6?[ ??!VB?W??4@)?-????1????n@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor* ?o_Ι?!G?߳i@) ?o_Ι?1G?߳i@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 51.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??[@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	⫄?????dW]?????_vO??!???{????	!       "	!       *	!       2$	}?F??P?????}n??гY?????!?ݓ??Z??:	!       B	!       J$	?I???p?ҁ?ɐ????{????!??<,Ԛ??R	!       Z$	?I???p?ҁ?ɐ????{????!??<,Ԛ??JCPU_ONLYY??[@b Y      Y@q?߄?;?D@"?
both?Your program is POTENTIALLY input-bound because 51.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?41.1659% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 