?($	B?c?]K????4??7???^)???!P??n???$	,?U ?@дZ()?@??6`@!???e۠)@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?-????a??+e??A?HP???Y=?U?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?):????jM????A????Y???B?i??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ڊ?e?????~?:??A??镲??Y0*??D??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&.?!??u??`vOj??A?镲q??Y?0?*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??????&S??:??AU0*????Y?j+??ݓ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&+??ݓ?????????A??H?}??Yc?ZB>???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????????o??Aa??+e??Y???{????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??~j?t??q???h ??A#J{?/L??Y?5?;Nё?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&/n????2U0*???A?St$????YM??St$??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	??ݓ???????H.??A8gDio??Y???H??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?Pk?w???]?Fx??A??	h"??Y???{????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&8gDio??[Ӽ???A????߾??Yvq?-??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&=
ףp=??????<,??A??????Y????<,??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???Mb???/?'??A?@??ǘ??Y?0?*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?G?z??      ??A?ʡE????Y?l??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????_v??Zd;?O???A???<,???Y?V-??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ݵ?|г??|a2U0??AT㥛? ??Y?? ?rh??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?H.?!???-C??6??A?f??j+??YV}??b??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??k	?????v??/??A?c]?F??Y??ܵ?|??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&P??n???8??d?`??Aj?q?????Yxz?,C??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?^)???W[??재?A?\?C????Y?U???؟?*	53333??@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?L?J???!<T???@@)؁sF????1V???>@:Preprocessing2F
Iterator::Model?c]?F??!?(???C@)[B>?٬??1Zs?DU:@:Preprocessing2U
Iterator::Model::ParallelMapV2l	??g???!`{???[*@)l	??g???1`{???[*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?&S???!??^n>N@)Qk?w????1j??$b#@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice? ?	???!I??{"@)? ?	???1I??{"@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::ConcatenateB?f??j??!xe??+@)?????K??1?4J?I?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?`TR'???!o??Hf1@)K?=?U??1@?N???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*Ǻ?????!ct???@)Ǻ?????1ct???@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 49.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9][?et-@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	b??0???޿?I??W[??재?!?v??/??	!       "	!       *	!       2$	?h??Ƥ???)?C?h???\?C????!j?q?????:	!       B	!       J$	?q>??? ??i2??vq?-??!xz?,C??R	!       Z$	?q>??? ??i2??vq?-??!xz?,C??JCPU_ONLYY][?et-@b Y      Y@q?{???A@"?
both?Your program is POTENTIALLY input-bound because 49.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?35.8882% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 